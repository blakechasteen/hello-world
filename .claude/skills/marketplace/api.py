"""
Skill Marketplace API
=====================

**Created**: November 22, 2025
**Purpose**: FastAPI backend for skill marketplace Web UI

This module provides REST endpoints for:
- Browsing and searching skills
- Installing, upgrading, and removing skills
- Real-time updates via WebSocket
- Statistics and analytics

## Architecture

```
React Frontend → FastAPI API → SkillCatalog/PackageManager
                    ↓
                WebSocket (real-time updates)
```

## Endpoints

**Catalog**:
- GET /api/skills - List all skills
- GET /api/skills/{skill_id} - Get skill details
- POST /api/skills/search - Search skills

**Package Management**:
- POST /api/skills/{skill_id}/install - Install skill
- POST /api/skills/{skill_id}/upgrade - Upgrade skill
- DELETE /api/skills/{skill_id} - Remove skill
- GET /api/skills/installed - List installed skills

**Analytics**:
- GET /api/stats - Marketplace statistics
- POST /api/skills/{skill_id}/rate - Rate a skill

**WebSocket**:
- WS /ws - Real-time updates
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from skills.marketplace.catalog import (
    SkillCatalog,
    SkillMetadata,
    SkillCategory,
    SortBy,
    SearchResult
)
from skills.marketplace.package_manager import (
    PackageManager,
    InstallResult,
    UpgradeResult,
    RemoveResult
)

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class SkillResponse(BaseModel):
    """Skill metadata for API responses."""
    skill_id: str
    name: str
    version: str
    category: str
    description: str
    tags: List[str]
    capabilities: List[str]

    # Requirements
    requires_warp_space: bool = False
    requires_yarn_graph: bool = False
    requires_event_bus: bool = False
    requires_mission_control: bool = False
    requires_tools: List[str] = []

    # Metadata
    author: str = ""
    homepage: str = ""
    repository: str = ""
    license: str = ""

    # Metrics
    install_count: int = 0
    usage_count: int = 0
    rating: float = 0.0
    rating_count: int = 0

    # Timestamps
    created_at: str
    updated_at: str
    last_used_at: Optional[str] = None

    @classmethod
    def from_metadata(cls, metadata: SkillMetadata) -> "SkillResponse":
        """Convert SkillMetadata to SkillResponse."""
        return cls(
            skill_id=metadata.skill_id,
            name=metadata.name,
            version=metadata.version,
            category=metadata.category.value,
            description=metadata.description,
            tags=metadata.tags,
            capabilities=metadata.capabilities,
            requires_warp_space=metadata.requires_warp_space,
            requires_yarn_graph=metadata.requires_yarn_graph,
            requires_event_bus=metadata.requires_event_bus,
            requires_mission_control=metadata.requires_mission_control,
            requires_tools=metadata.requires_tools,
            author=metadata.author,
            homepage=metadata.homepage,
            repository=metadata.repository,
            license=metadata.license,
            install_count=metadata.install_count,
            usage_count=metadata.usage_count,
            rating=metadata.rating,
            rating_count=metadata.rating_count,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            last_used_at=metadata.last_used_at
        )


class SearchRequest(BaseModel):
    """Search request."""
    query: str
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = Field(default=20, ge=1, le=100)


class SearchResultResponse(BaseModel):
    """Search result."""
    skill: SkillResponse
    score: float
    matched_fields: List[str]


class InstallRequest(BaseModel):
    """Installation request."""
    force: bool = False
    skip_tests: bool = False
    skip_dependencies: bool = False


class InstallResponse(BaseModel):
    """Installation response."""
    success: bool
    skill_id: str
    message: str
    installed_path: Optional[str] = None
    dependencies_installed: List[str] = []
    warnings: List[str] = []
    errors: List[str] = []


class UpgradeRequest(BaseModel):
    """Upgrade request."""
    version: Optional[str] = None
    skip_tests: bool = False


class UpgradeResponse(BaseModel):
    """Upgrade response."""
    success: bool
    skill_id: str
    message: str
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    warnings: List[str] = []
    errors: List[str] = []


class RemoveRequest(BaseModel):
    """Remove request."""
    force: bool = False


class RemoveResponse(BaseModel):
    """Remove response."""
    success: bool
    skill_id: str
    message: str
    backup_path: Optional[str] = None
    warnings: List[str] = []
    errors: List[str] = []


class RatingRequest(BaseModel):
    """Rating request."""
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class StatsResponse(BaseModel):
    """Marketplace statistics."""
    total_skills: int
    by_category: Dict[str, int]
    total_installs: int
    total_usage: int
    avg_rating: float


class WebSocketMessage(BaseModel):
    """WebSocket message."""
    type: str  # "install_started", "install_progress", "install_complete", etc.
    skill_id: str
    data: Dict[str, Any]
    timestamp: str


# ============================================================================
# WebSocket Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept new connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove connection."""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Active: {len(self.active_connections)}")

    async def broadcast(self, message: WebSocketMessage):
        """Broadcast message to all connections."""
        message_json = message.model_dump_json()
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="HoloLoom Skill Marketplace API",
    description="REST API for skill discovery, installation, and management",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
catalog: Optional[SkillCatalog] = None
package_manager: Optional[PackageManager] = None
ws_manager = ConnectionManager()


# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize catalog and package manager."""
    global catalog, package_manager

    logger.info("Initializing Skill Marketplace API...")

    # Initialize catalog
    catalog = SkillCatalog()
    await catalog.initialize()

    # Initialize package manager
    package_manager = PackageManager(catalog)

    logger.info("Skill Marketplace API ready")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup."""
    global catalog

    logger.info("Shutting down Skill Marketplace API...")

    if catalog:
        await catalog.close()

    logger.info("Skill Marketplace API shutdown complete")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "catalog_ready": catalog is not None,
        "package_manager_ready": package_manager is not None
    }


# ============================================================================
# Catalog Endpoints
# ============================================================================

@app.get("/api/skills", response_model=List[SkillResponse])
async def list_skills(
    category: Optional[str] = None,
    sort_by: str = "name",
    limit: Optional[int] = None
):
    """List all skills."""
    if not catalog:
        raise HTTPException(status_code=503, detail="Catalog not initialized")

    # Parse category
    cat = SkillCategory(category) if category else None

    # Parse sort_by
    try:
        sort = SortBy(sort_by)
    except ValueError:
        sort = SortBy.NAME

    # List skills
    skills = await catalog.list_all(category=cat, sort_by=sort, limit=limit)

    return [SkillResponse.from_metadata(s) for s in skills]


@app.get("/api/skills/{skill_id}", response_model=SkillResponse)
async def get_skill(skill_id: str):
    """Get skill details."""
    if not catalog:
        raise HTTPException(status_code=503, detail="Catalog not initialized")

    skill = await catalog.get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_id}")

    return SkillResponse.from_metadata(skill)


@app.post("/api/skills/search", response_model=List[SearchResultResponse])
async def search_skills(request: SearchRequest):
    """Search skills."""
    if not catalog:
        raise HTTPException(status_code=503, detail="Catalog not initialized")

    # Parse category
    cat = SkillCategory(request.category) if request.category else None

    # Search
    results = await catalog.search(
        query=request.query,
        category=cat,
        tags=request.tags,
        limit=request.limit
    )

    return [
        SearchResultResponse(
            skill=SkillResponse.from_metadata(r.skill),
            score=r.score,
            matched_fields=r.matched_fields
        )
        for r in results
    ]


# ============================================================================
# Package Management Endpoints
# ============================================================================

@app.post("/api/skills/{skill_id}/install", response_model=InstallResponse)
async def install_skill(skill_id: str, request: InstallRequest):
    """Install a skill."""
    if not package_manager:
        raise HTTPException(status_code=503, detail="Package manager not initialized")

    # Broadcast start
    await ws_manager.broadcast(WebSocketMessage(
        type="install_started",
        skill_id=skill_id,
        data={},
        timestamp=datetime.now().isoformat()
    ))

    # Install
    result = await package_manager.install(
        skill_id=skill_id,
        force=request.force,
        skip_tests=request.skip_tests,
        skip_dependencies=request.skip_dependencies
    )

    # Broadcast complete
    await ws_manager.broadcast(WebSocketMessage(
        type="install_complete",
        skill_id=skill_id,
        data={"success": result.success},
        timestamp=datetime.now().isoformat()
    ))

    return InstallResponse(
        success=result.success,
        skill_id=result.skill_id,
        message=result.message,
        installed_path=str(result.installed_path) if result.installed_path else None,
        dependencies_installed=result.dependencies_installed,
        warnings=result.warnings,
        errors=result.errors
    )


@app.post("/api/skills/{skill_id}/upgrade", response_model=UpgradeResponse)
async def upgrade_skill(skill_id: str, request: UpgradeRequest):
    """Upgrade a skill."""
    if not package_manager:
        raise HTTPException(status_code=503, detail="Package manager not initialized")

    # Broadcast start
    await ws_manager.broadcast(WebSocketMessage(
        type="upgrade_started",
        skill_id=skill_id,
        data={},
        timestamp=datetime.now().isoformat()
    ))

    # Upgrade
    result = await package_manager.upgrade(
        skill_id=skill_id,
        version=request.version,
        skip_tests=request.skip_tests
    )

    # Broadcast complete
    await ws_manager.broadcast(WebSocketMessage(
        type="upgrade_complete",
        skill_id=skill_id,
        data={"success": result.success},
        timestamp=datetime.now().isoformat()
    ))

    return UpgradeResponse(
        success=result.success,
        skill_id=result.skill_id,
        message=result.message,
        old_version=result.old_version,
        new_version=result.new_version,
        warnings=result.warnings,
        errors=result.errors
    )


@app.delete("/api/skills/{skill_id}", response_model=RemoveResponse)
async def remove_skill(skill_id: str, request: RemoveRequest):
    """Remove a skill."""
    if not package_manager:
        raise HTTPException(status_code=503, detail="Package manager not initialized")

    # Broadcast start
    await ws_manager.broadcast(WebSocketMessage(
        type="remove_started",
        skill_id=skill_id,
        data={},
        timestamp=datetime.now().isoformat()
    ))

    # Remove
    result = await package_manager.remove(
        skill_id=skill_id,
        force=request.force
    )

    # Broadcast complete
    await ws_manager.broadcast(WebSocketMessage(
        type="remove_complete",
        skill_id=skill_id,
        data={"success": result.success},
        timestamp=datetime.now().isoformat()
    ))

    return RemoveResponse(
        success=result.success,
        skill_id=result.skill_id,
        message=result.message,
        backup_path=str(result.backup_path) if result.backup_path else None,
        warnings=result.warnings,
        errors=result.errors
    )


@app.get("/api/skills/installed", response_model=List[SkillResponse])
async def list_installed():
    """List installed skills."""
    if not package_manager:
        raise HTTPException(status_code=503, detail="Package manager not initialized")

    skills = await package_manager.list_installed()

    return [SkillResponse.from_metadata(s) for s in skills]


@app.get("/api/skills/updates")
async def check_updates():
    """Check for available updates."""
    if not package_manager:
        raise HTTPException(status_code=503, detail="Package manager not initialized")

    updates = await package_manager.check_updates()

    return {
        "updates_available": len(updates),
        "updates": updates
    }


# ============================================================================
# Analytics Endpoints
# ============================================================================

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get marketplace statistics."""
    if not catalog:
        raise HTTPException(status_code=503, detail="Catalog not initialized")

    stats = await catalog.get_stats()

    return StatsResponse(**stats)


@app.post("/api/skills/{skill_id}/rate")
async def rate_skill(skill_id: str, request: RatingRequest):
    """Rate a skill."""
    if not catalog:
        raise HTTPException(status_code=503, detail="Catalog not initialized")

    # Add rating
    await catalog.add_rating(
        skill_id=skill_id,
        rating=request.rating,
        comment=request.comment
    )

    # Get updated skill
    skill = await catalog.get_skill(skill_id)

    return {
        "success": True,
        "skill_id": skill_id,
        "new_rating": skill.rating if skill else 0.0,
        "rating_count": skill.rating_count if skill else 0
    }


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await ws_manager.connect(websocket)

    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()

            # Echo back (for ping/pong)
            await websocket.send_text(data)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

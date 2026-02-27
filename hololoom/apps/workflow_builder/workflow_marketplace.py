"""
Workflow Marketplace API Endpoints.

Provides REST API for browsing, searching, and using workflow templates
from the marketplace.

Endpoints:
    GET  /api/marketplace/categories           - List all categories
    GET  /api/marketplace/templates            - List templates (paginated)
    GET  /api/marketplace/templates/featured   - Get featured templates
    GET  /api/marketplace/templates/search     - Search templates
    GET  /api/marketplace/templates/{id}       - Get template details
    POST /api/marketplace/templates/{id}/use   - Track template usage
    POST /api/marketplace/templates/{id}/rate  - Submit rating
    GET  /api/marketplace/templates/{id}/export - Export template JSON

Usage:
    from hololoom.apps.workflow_builder.workflow_marketplace import create_marketplace_router

    app = FastAPI()
    router = await create_marketplace_router("./workflows.db")
    app.include_router(router, prefix="/api/marketplace")
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class CategoryResponse(BaseModel):
    """Category in the marketplace."""
    id: str
    name: str
    parent_id: Optional[str] = None
    description: str
    icon: str
    display_order: int


class TemplateListItem(BaseModel):
    """Template summary for list views."""
    id: str
    workflow_id: str
    name: str
    description: str
    category: str
    difficulty: str
    author: str
    is_featured: bool
    usage_count: int
    avg_rating: float
    rating_count: int
    tags: List[str]


class TemplateDetail(BaseModel):
    """Full template details including workflow definition."""
    id: str
    workflow_id: str
    name: str
    description: str
    category: str
    difficulty: str
    author: str
    license: str
    is_featured: bool
    usage_count: int
    avg_rating: float
    rating_count: int
    tags: List[str]
    published_at: str
    updated_at: str
    # Workflow details
    nodes: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class TemplateListResponse(BaseModel):
    """Paginated template list response."""
    templates: List[TemplateListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class RatingRequest(BaseModel):
    """Request to rate a template."""
    user_id: str = Field(..., description="User submitting the rating")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: str = Field("", description="Optional comment")


class RatingResponse(BaseModel):
    """Response after submitting a rating."""
    success: bool
    message: str
    new_avg_rating: float
    new_rating_count: int


class UsageResponse(BaseModel):
    """Response after tracking template usage."""
    success: bool
    message: str
    new_usage_count: int


class ExportResponse(BaseModel):
    """Exported workflow JSON."""
    workflow: Dict[str, Any]
    marketplace_metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Router Factory
# ---------------------------------------------------------------------------

async def create_marketplace_router(db_path: str = "./workflows.db") -> APIRouter:
    """
    Create the marketplace API router.

    Args:
        db_path: Path to the SQLite database

    Returns:
        FastAPI router with marketplace endpoints
    """
    from .workflow_persistence import WorkflowPersistence

    router = APIRouter(tags=["Marketplace"])
    persistence = WorkflowPersistence(db_path)
    await persistence.initialize()

    # -----------------------------------------------------------------------
    # Categories
    # -----------------------------------------------------------------------

    @router.get("/categories", response_model=List[CategoryResponse])
    async def list_categories():
        """Get all template categories."""
        categories = await persistence.get_all_categories()
        return [
            CategoryResponse(
                id=c.id,
                name=c.name,
                parent_id=c.parent_id,
                description=c.description,
                icon=c.icon,
                display_order=c.display_order
            )
            for c in categories
        ]

    # -----------------------------------------------------------------------
    # Templates - List & Search
    # -----------------------------------------------------------------------

    @router.get("/templates", response_model=TemplateListResponse)
    async def list_templates(
        category: Optional[str] = Query(None, description="Filter by category"),
        difficulty: Optional[str] = Query(None, description="Filter by difficulty"),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page")
    ):
        """List marketplace templates with optional filtering and pagination."""
        # Get all matching templates
        templates = await persistence.list_marketplace_templates(
            category=category,
            difficulty=difficulty,
            featured_only=False
        )

        # Enrich with workflow names
        enriched = []
        for t in templates:
            workflow = await persistence.get_workflow(t.workflow_id)
            if workflow:
                enriched.append(TemplateListItem(
                    id=t.id,
                    workflow_id=t.workflow_id,
                    name=workflow.name,
                    description=workflow.description,
                    category=t.category,
                    difficulty=t.difficulty,
                    author=t.author,
                    is_featured=t.is_featured,
                    usage_count=t.usage_count,
                    avg_rating=t.avg_rating,
                    rating_count=t.rating_count,
                    tags=t.tags
                ))

        # Paginate
        total = len(enriched)
        total_pages = (total + page_size - 1) // page_size
        start = (page - 1) * page_size
        end = start + page_size
        page_items = enriched[start:end]

        return TemplateListResponse(
            templates=page_items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    @router.get("/templates/featured", response_model=List[TemplateListItem])
    async def get_featured_templates():
        """Get featured templates for homepage display."""
        templates = await persistence.list_marketplace_templates(featured_only=True)

        enriched = []
        for t in templates:
            workflow = await persistence.get_workflow(t.workflow_id)
            if workflow:
                enriched.append(TemplateListItem(
                    id=t.id,
                    workflow_id=t.workflow_id,
                    name=workflow.name,
                    description=workflow.description,
                    category=t.category,
                    difficulty=t.difficulty,
                    author=t.author,
                    is_featured=t.is_featured,
                    usage_count=t.usage_count,
                    avg_rating=t.avg_rating,
                    rating_count=t.rating_count,
                    tags=t.tags
                ))

        return enriched

    @router.get("/templates/search", response_model=List[TemplateListItem])
    async def search_templates(
        q: str = Query(..., min_length=1, description="Search query"),
        category: Optional[str] = Query(None, description="Filter by category"),
        difficulty: Optional[str] = Query(None, description="Filter by difficulty"),
        limit: int = Query(20, ge=1, le=100, description="Max results")
    ):
        """Search templates by name, description, or tags."""
        # Search in workflows
        templates = await persistence.search_marketplace_templates(
            query=q,
            category=category,
            difficulty=difficulty,
            limit=limit
        )

        enriched = []
        for t in templates:
            workflow = await persistence.get_workflow(t.workflow_id)
            if workflow:
                enriched.append(TemplateListItem(
                    id=t.id,
                    workflow_id=t.workflow_id,
                    name=workflow.name,
                    description=workflow.description,
                    category=t.category,
                    difficulty=t.difficulty,
                    author=t.author,
                    is_featured=t.is_featured,
                    usage_count=t.usage_count,
                    avg_rating=t.avg_rating,
                    rating_count=t.rating_count,
                    tags=t.tags
                ))

        return enriched

    # -----------------------------------------------------------------------
    # Templates - Single Item Operations
    # -----------------------------------------------------------------------

    @router.get("/templates/{template_id}", response_model=TemplateDetail)
    async def get_template(template_id: str):
        """Get full details for a single template."""
        template = await persistence.get_marketplace_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        workflow = await persistence.get_workflow(template.workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return TemplateDetail(
            id=template.id,
            workflow_id=template.workflow_id,
            name=workflow.name,
            description=workflow.description,
            category=template.category,
            difficulty=template.difficulty,
            author=template.author,
            license=template.license,
            is_featured=template.is_featured,
            usage_count=template.usage_count,
            avg_rating=template.avg_rating,
            rating_count=template.rating_count,
            tags=template.tags,
            published_at=template.published_at.isoformat(),
            updated_at=template.updated_at.isoformat(),
            nodes=workflow.nodes,
            connections=workflow.connections,
            metadata=workflow.metadata
        )

    @router.post("/templates/{template_id}/use", response_model=UsageResponse)
    async def track_template_usage(template_id: str):
        """Track when a user imports/uses a template."""
        template = await persistence.get_marketplace_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        new_count = await persistence.increment_usage_count(template_id)

        return UsageResponse(
            success=True,
            message="Usage tracked",
            new_usage_count=new_count
        )

    @router.post("/templates/{template_id}/rate", response_model=RatingResponse)
    async def rate_template(template_id: str, request: RatingRequest):
        """Submit a rating for a template."""
        template = await persistence.get_marketplace_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        from .workflow_persistence import TemplateRatingRecord
        import uuid

        rating = TemplateRatingRecord(
            id=str(uuid.uuid4()),
            template_id=template_id,
            user_id=request.user_id,
            rating=request.rating,
            comment=request.comment,
            created_at=datetime.utcnow()
        )

        await persistence.save_rating(rating)

        # Get updated template to return new stats
        updated = await persistence.get_marketplace_template(template_id)

        return RatingResponse(
            success=True,
            message="Rating submitted",
            new_avg_rating=updated.avg_rating if updated else 0.0,
            new_rating_count=updated.rating_count if updated else 0
        )

    @router.get("/templates/{template_id}/export", response_model=ExportResponse)
    async def export_template(template_id: str):
        """Export a template as importable JSON."""
        template = await persistence.get_marketplace_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        workflow = await persistence.get_workflow(template.workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Build standard workflow JSON
        workflow_json = {
            "version": workflow.metadata.get("version", "1.0"),
            "name": workflow.name,
            "description": workflow.description,
            "nodes": workflow.nodes,
            "connections": workflow.connections
        }

        # Add any extra metadata
        for key, value in workflow.metadata.items():
            if key not in ("version",):
                workflow_json[key] = value

        # Marketplace metadata
        marketplace_meta = {
            "marketplace_id": template.id,
            "category": template.category,
            "difficulty": template.difficulty,
            "author": template.author,
            "license": template.license,
            "tags": template.tags,
            "usage_count": template.usage_count,
            "avg_rating": template.avg_rating,
            "rating_count": template.rating_count,
            "exported_at": datetime.utcnow().isoformat()
        }

        return ExportResponse(
            workflow=workflow_json,
            marketplace_metadata=marketplace_meta
        )

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    @router.get("/stats")
    async def get_marketplace_stats():
        """Get overall marketplace statistics."""
        categories = await persistence.get_all_categories()
        all_templates = await persistence.list_marketplace_templates()
        featured = await persistence.list_marketplace_templates(featured_only=True)

        # Calculate stats
        total_usage = sum(t.usage_count for t in all_templates)
        avg_rating = (
            sum(t.avg_rating * t.rating_count for t in all_templates if t.rating_count > 0) /
            sum(t.rating_count for t in all_templates if t.rating_count > 0)
            if any(t.rating_count > 0 for t in all_templates)
            else 0.0
        )

        # By category
        by_category = {}
        for t in all_templates:
            if t.category not in by_category:
                by_category[t.category] = {"count": 0, "usage": 0}
            by_category[t.category]["count"] += 1
            by_category[t.category]["usage"] += t.usage_count

        # By difficulty
        by_difficulty = {"beginner": 0, "intermediate": 0, "advanced": 0}
        for t in all_templates:
            if t.difficulty in by_difficulty:
                by_difficulty[t.difficulty] += 1

        return {
            "total_templates": len(all_templates),
            "total_categories": len(categories),
            "featured_count": len(featured),
            "total_usage": total_usage,
            "average_rating": round(avg_rating, 2),
            "by_category": by_category,
            "by_difficulty": by_difficulty
        }

    return router


# ---------------------------------------------------------------------------
# Standalone Server (for testing)
# ---------------------------------------------------------------------------

async def run_standalone(db_path: str = "./workflows.db", port: int = 8002):
    """Run the marketplace API as a standalone server."""
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(
        title="HoloLoom Workflow Marketplace",
        description="Browse, search, and import workflow templates",
        version="1.0.0"
    )

    router = await create_marketplace_router(db_path)
    app.include_router(router, prefix="/api/marketplace")

    # Add CORS for frontend
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {
            "service": "HoloLoom Workflow Marketplace",
            "docs": "/docs",
            "endpoints": {
                "categories": "/api/marketplace/categories",
                "templates": "/api/marketplace/templates",
                "featured": "/api/marketplace/templates/featured",
                "search": "/api/marketplace/templates/search?q=<query>",
                "stats": "/api/marketplace/stats"
            }
        }

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def main():
    """CLI entry point."""
    import asyncio
    import sys

    db_path = "./workflows.db"
    port = 8002

    # Parse args
    if "--db-path" in sys.argv:
        idx = sys.argv.index("--db-path")
        if idx + 1 < len(sys.argv):
            db_path = sys.argv[idx + 1]

    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        print("\nOptions:")
        print("  --db-path PATH   Database path (default: ./workflows.db)")
        print("  --port PORT      Server port (default: 8002)")
        return

    print(f"Starting Marketplace API on port {port}...")
    print(f"Database: {db_path}")
    print(f"Docs: http://localhost:{port}/docs")

    asyncio.run(run_standalone(db_path, port))


if __name__ == "__main__":
    main()

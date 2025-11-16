#!/usr/bin/env python3
"""
HoloLoom Promptly Real-Time Dashboard Server
============================================

Real-time WebSocket dashboard for visualizing:
- Memory graph (knowledge graph nodes and edges)
- Recursive reasoning metrics (strategy performance)
- Skill execution tracking
- Analytics trends

Features:
- WebSocket for real-time updates
- REST API (v1) with rate limiting
- FastAPI backend
- Connects to analytics database
- Live monitoring of all Promptly integration features

Usage:
    uvicorn HoloLoom.dashboard_server:app --reload --port 8000

Then open: http://localhost:8000

Created: 2025-11-16
Updated: 2025-11-16 (Added rate limiting and API versioning)
Integration: Phases 1-4 → Real-time Dashboard
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Try to import slowapi for rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    Limiter = None

# HoloLoom imports
from HoloLoom.config import Config
from HoloLoom.analytics.recursive_analytics import RecursiveAnalytics
from HoloLoom.weaving_orchestrator_recursive import RecursiveWeavingOrchestrator
from HoloLoom.agentic.skill_agents import SkillRegistry, list_available_skills

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Simple Rate Limiter (fallback if slowapi not available)
# ============================================================================

class SimpleRateLimiter:
    """Simple in-memory rate limiter as fallback."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]

        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Record request
        self.requests[client_id].append(now)
        return True

    async def __call__(self, request: Request):
        """Middleware-style rate limit check."""
        client_id = request.client.host if request.client else "unknown"

        if not self.check_rate_limit(client_id):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds} seconds."
            )


# ============================================================================
# FastAPI App Initialization with Rate Limiting
# ============================================================================

# Create FastAPI app with API versioning
app = FastAPI(
    title="HoloLoom Promptly Dashboard",
    version="1.0.0",
    description="Real-time dashboard with v1 REST API and rate limiting"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter
if RATE_LIMITING_AVAILABLE:
    logger.info("Using slowapi for rate limiting")
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    logger.warning("slowapi not available - using fallback rate limiter. Install with: pip install slowapi")
    limiter = SimpleRateLimiter(max_requests=100, window_seconds=60)

# Global state
analytics: Optional[RecursiveAnalytics] = None
skill_registry: Optional[SkillRegistry] = None
active_websockets: Set[WebSocket] = set()
config: Optional[Config] = None


# ============================================================================
# Initialization
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize dashboard components."""
    global analytics, skill_registry, config

    logger.info("Initializing HoloLoom Promptly Dashboard...")

    # Load configuration
    config = Config.fast()
    logger.info(f"Configuration: {config.mode.value} mode")

    # Load analytics
    analytics = RecursiveAnalytics()
    logger.info("Analytics database connected")

    # Load skill registry
    skill_registry = SkillRegistry()
    await skill_registry.load_all_skills()
    logger.info(f"Loaded {len(skill_registry.skills)} professional skills")

    logger.info("Dashboard server ready! API version: v1, Rate limiting: enabled")


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.add(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# ============================================================================
# WebSocket Endpoint (no rate limiting)
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        # Send initial data
        await send_initial_data(websocket)

        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "request_update":
                await send_analytics_update(websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def send_initial_data(websocket: WebSocket):
    """Send initial dashboard data to new connection."""
    # Analytics summary
    summary = analytics.get_summary()

    # Available skills
    skills = await list_available_skills()

    # Recent executions (last 10)
    recent = analytics.get_recent_executions(limit=10)

    initial_data = {
        "type": "initial",
        "analytics": summary,
        "skills": skills,
        "recent_executions": recent
    }

    await websocket.send_json(initial_data)


async def send_analytics_update(websocket: WebSocket):
    """Send analytics update to client."""
    summary = analytics.get_summary()

    update = {
        "type": "analytics_update",
        "data": summary,
        "timestamp": datetime.now().isoformat()
    }

    await websocket.send_json(update)


# ============================================================================
# REST API Endpoints (v1 with rate limiting)
# ============================================================================

@app.get("/")
async def get_dashboard():
    """Serve dashboard HTML."""
    dashboard_html = Path(__file__).parent / "dashboard.html"

    if dashboard_html.exists():
        return FileResponse(dashboard_html)
    else:
        return HTMLResponse(content=get_embedded_dashboard_html())


# API v1 endpoints with rate limiting
if RATE_LIMITING_AVAILABLE:
    @app.get("/api/v1/analytics/summary")
    @limiter.limit("100/minute")
    async def get_analytics_summary(request: Request):
        """Get analytics summary."""
        summary = analytics.get_summary()
        return summary

    @app.get("/api/v1/analytics/trends")
    @limiter.limit("100/minute")
    async def get_analytics_trends(
        request: Request,
        days: int = 7,
        skip: int = 0,
        limit: int = 50
    ):
        """
        Get quality trends over time with optional pagination.

        Parameters:
        - days: Number of days to look back (default: 7)
        - skip: Number of items to skip (default: 0)
        - limit: Number of items per page (default: 50, max: 100)

        Response includes pagination metadata when limit is specified.
        """
        # Validate pagination parameters
        if skip < 0:
            raise HTTPException(status_code=400, detail="skip must be >= 0")
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")

        # Cap limit at reasonable max
        max_limit = 100
        if limit > max_limit:
            limit = max_limit

        # Get all trends
        all_trends = analytics.get_quality_trends(days=days)

        # Calculate pagination metadata
        total = len(all_trends)
        has_more = (skip + limit) < total
        next_skip = skip + limit if has_more else None

        # Apply pagination
        paginated_trends = all_trends[skip:skip + limit]

        return {
            "trends": paginated_trends,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": total,
                "count": len(paginated_trends),
                "has_more": has_more,
                "next_skip": next_skip
            }
        }

    @app.get("/api/v1/analytics/strategy/{strategy}")
    @limiter.limit("100/minute")
    async def get_strategy_metrics(request: Request, strategy: str):
        """Get metrics for specific strategy."""
        metrics = analytics.get_strategy_metrics(strategy)

        if not metrics:
            raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy}")

        return {
            "strategy": strategy,
            "total_executions": metrics.total_executions,
            "avg_iterations": metrics.avg_iterations,
            "avg_quality_gain": metrics.avg_quality_gain,
            "avg_duration_ms": metrics.avg_duration_ms,
            "success_rate": metrics.success_rate,
            "convergence_rate": metrics.convergence_rate
        }

    @app.get("/api/v1/analytics/recommendations")
    @limiter.limit("100/minute")
    async def get_recommendations(request: Request):
        """Get AI-powered recommendations."""
        recommendations = analytics.get_recommendations()
        return {"recommendations": recommendations}

    @app.get("/api/v1/skills")
    @limiter.limit("100/minute")
    async def get_skills(request: Request):
        """Get all available skills."""
        skills = await list_available_skills()
        return skills

    @app.get("/api/v1/skills/{skill_name}")
    @limiter.limit("100/minute")
    async def get_skill_details(request: Request, skill_name: str):
        """Get details for specific skill."""
        skill = skill_registry.get_skill(skill_name)

        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

        return {
            "name": skill.name,
            "version": skill.version,
            "description": skill.description,
            "category": skill.metadata.category,
            "tags": skill.metadata.tags,
            "strategy": skill.reasoning.default_strategy,
            "max_iterations": skill.reasoning.max_iterations,
            "quality_threshold": skill.reasoning.quality_threshold,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "description": p.description
                }
                for p in skill.parameters
            ]
        }

    @app.get("/api/v1/executions/recent")
    @limiter.limit("100/minute")
    async def get_recent_executions(
        request: Request,
        skip: int = 0,
        limit: int = 20
    ):
        """
        Get recent executions with pagination support.

        Parameters:
        - skip: Number of items to skip (default: 0)
        - limit: Number of items per page (default: 20, max: 100)

        Response includes pagination metadata:
        - skip: Current offset
        - limit: Current page size
        - total: Total number of items
        - count: Number of items in current page
        - has_more: Whether more pages exist
        - next_skip: Skip value for next page (null if last page)
        """
        # Validate pagination parameters
        if skip < 0:
            raise HTTPException(status_code=400, detail="skip must be >= 0")
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")

        # Cap limit at reasonable max to prevent abuse
        max_limit = 100
        if limit > max_limit:
            limit = max_limit

        # Get total count for pagination metadata
        total = analytics.get_total_executions_count()

        # Fetch paginated results
        recent = analytics.get_recent_executions(limit=limit, skip=skip)

        # Convert ExecutionRecord objects to dictionaries
        executions_data = [record.to_dict() for record in recent]

        # Calculate pagination metadata
        has_more = (skip + limit) < total
        next_skip = skip + limit if has_more else None

        return {
            "executions": executions_data,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": total,
                "count": len(executions_data),
                "has_more": has_more,
                "next_skip": next_skip
            }
        }

else:
    # Fallback endpoints with simple rate limiting
    @app.get("/api/v1/analytics/summary")
    async def get_analytics_summary(request: Request):
        """Get analytics summary."""
        await limiter(request)
        summary = analytics.get_summary()
        return summary

    @app.get("/api/v1/analytics/trends")
    async def get_analytics_trends(
        request: Request,
        days: int = 7,
        skip: int = 0,
        limit: int = 50
    ):
        """
        Get quality trends over time with optional pagination.

        Parameters:
        - days: Number of days to look back (default: 7)
        - skip: Number of items to skip (default: 0)
        - limit: Number of items per page (default: 50, max: 100)

        Response includes pagination metadata when limit is specified.
        """
        await limiter(request)

        # Validate pagination parameters
        if skip < 0:
            raise HTTPException(status_code=400, detail="skip must be >= 0")
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")

        # Cap limit at reasonable max
        max_limit = 100
        if limit > max_limit:
            limit = max_limit

        # Get all trends
        all_trends = analytics.get_quality_trends(days=days)

        # Calculate pagination metadata
        total = len(all_trends)
        has_more = (skip + limit) < total
        next_skip = skip + limit if has_more else None

        # Apply pagination
        paginated_trends = all_trends[skip:skip + limit]

        return {
            "trends": paginated_trends,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": total,
                "count": len(paginated_trends),
                "has_more": has_more,
                "next_skip": next_skip
            }
        }

    @app.get("/api/v1/analytics/strategy/{strategy}")
    async def get_strategy_metrics(request: Request, strategy: str):
        """Get metrics for specific strategy."""
        await limiter(request)
        metrics = analytics.get_strategy_metrics(strategy)

        if not metrics:
            raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy}")

        return {
            "strategy": strategy,
            "total_executions": metrics.total_executions,
            "avg_iterations": metrics.avg_iterations,
            "avg_quality_gain": metrics.avg_quality_gain,
            "avg_duration_ms": metrics.avg_duration_ms,
            "success_rate": metrics.success_rate,
            "convergence_rate": metrics.convergence_rate
        }

    @app.get("/api/v1/analytics/recommendations")
    async def get_recommendations(request: Request):
        """Get AI-powered recommendations."""
        await limiter(request)
        recommendations = analytics.get_recommendations()
        return {"recommendations": recommendations}

    @app.get("/api/v1/skills")
    async def get_skills(request: Request):
        """Get all available skills."""
        await limiter(request)
        skills = await list_available_skills()
        return skills

    @app.get("/api/v1/skills/{skill_name}")
    async def get_skill_details(request: Request, skill_name: str):
        """Get details for specific skill."""
        await limiter(request)
        skill = skill_registry.get_skill(skill_name)

        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

        return {
            "name": skill.name,
            "version": skill.version,
            "description": skill.description,
            "category": skill.metadata.category,
            "tags": skill.metadata.tags,
            "strategy": skill.reasoning.default_strategy,
            "max_iterations": skill.reasoning.max_iterations,
            "quality_threshold": skill.reasoning.quality_threshold,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "description": p.description
                }
                for p in skill.parameters
            ]
        }

    @app.get("/api/v1/executions/recent")
    async def get_recent_executions(
        request: Request,
        skip: int = 0,
        limit: int = 20
    ):
        """
        Get recent executions with pagination support.

        Parameters:
        - skip: Number of items to skip (default: 0)
        - limit: Number of items per page (default: 20, max: 100)

        Response includes pagination metadata:
        - skip: Current offset
        - limit: Current page size
        - total: Total number of items
        - count: Number of items in current page
        - has_more: Whether more pages exist
        - next_skip: Skip value for next page (null if last page)
        """
        await limiter(request)

        # Validate pagination parameters
        if skip < 0:
            raise HTTPException(status_code=400, detail="skip must be >= 0")
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")

        # Cap limit at reasonable max to prevent abuse
        max_limit = 100
        if limit > max_limit:
            limit = max_limit

        # Get total count for pagination metadata
        total = analytics.get_total_executions_count()

        # Fetch paginated results
        recent = analytics.get_recent_executions(limit=limit, skip=skip)

        # Convert ExecutionRecord objects to dictionaries
        executions_data = [record.to_dict() for record in recent]

        # Calculate pagination metadata
        has_more = (skip + limit) < total
        next_skip = skip + limit if has_more else None

        return {
            "executions": executions_data,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": total,
                "count": len(executions_data),
                "has_more": has_more,
                "next_skip": next_skip
            }
        }


# ============================================================================
# Background Tasks
# ============================================================================

async def broadcast_analytics_updates():
    """Background task to broadcast analytics updates."""
    while True:
        await asyncio.sleep(5)  # Update every 5 seconds

        if manager.active_connections:
            summary = analytics.get_summary()

            update = {
                "type": "analytics_update",
                "data": summary,
                "timestamp": datetime.now().isoformat()
            }

            await manager.broadcast(update)


@app.on_event("startup")
async def start_background_tasks():
    """Start background update tasks."""
    asyncio.create_task(broadcast_analytics_updates())


# ============================================================================
# Embedded Dashboard HTML (fallback)
# ============================================================================

def get_embedded_dashboard_html() -> str:
    """Get embedded dashboard HTML with API v1 endpoints."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>HoloLoom Promptly Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .status {
            background: #1a1f3a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #00d4ff;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: #1a1f3a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #2a3555;
        }
        .card h2 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #2a3555;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #888; }
        .metric-value {
            color: #00ff88;
            font-weight: 600;
        }
        .connection-status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .connected {
            background: #00ff8820;
            color: #00ff88;
        }
        .disconnected {
            background: #ff440020;
            color: #ff4400;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }

        /* Sticky table header */
        thead {
            position: sticky;
            top: 0;
            background: #1a1f3a;
            z-index: 10;
        }

        th, td {
            padding: 12px 10px;
            text-align: left;
            border-bottom: 1px solid #2a3555;
        }

        th {
            color: #00d4ff;
            font-weight: 600;
        }

        /* Zebra striping for better scannability */
        tbody tr {
            background: #0a0e27;
            transition: background-color 0.2s ease;
        }

        tbody tr:nth-child(even) {
            background: #151833;
        }

        /* Hover state for interactive feedback */
        tbody tr:hover {
            background: #2a3555;
            cursor: pointer;
        }

        /* First and last column alignment polish */
        td:first-child, th:first-child {
            padding-left: 15px;
        }

        td:last-child, th:last-child {
            padding-right: 15px;
            text-align: right;
        }

        .timestamp {
            color: #666;
            font-size: 12px;
        }

        /* Loading Skeleton Styles */
        .skeleton {
            background: linear-gradient(90deg, #1a1f3a 25%, #2a3555 50%, #1a1f3a 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 4px;
            height: 20px;
            margin: 5px 0;
        }

        .skeleton-text {
            width: 60%;
        }

        .skeleton-number {
            width: 40%;
        }

        .skeleton-item {
            height: 16px;
            margin-bottom: 10px;
            border-radius: 4px;
            background: linear-gradient(90deg, #1a1f3a 25%, #2a3555 50%, #1a1f3a 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        }

        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        /* Tooltip Styles */
        .metric-label {
            position: relative;
            cursor: help;
            border-bottom: 1px dotted #555;
        }

        .metric-label:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: #0a0e27;
            color: #00d4ff;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            white-space: nowrap;
            border: 1px solid #2a3555;
            box-shadow: 0 4px 12px rgba(0, 212, 255, 0.15);
            z-index: 1000;
            pointer-events: none;
        }

        .metric-label:hover::before {
            content: '';
            position: absolute;
            bottom: 115%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: #2a3555;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 HoloLoom Promptly Dashboard</h1>
        <div class="status">
            <strong>WebSocket:</strong> <span id="ws-status" class="connection-status disconnected">Disconnected</span>
            <span style="margin-left: 20px; color: #666;">Last update: <span id="last-update">Never</span></span>
            <span style="margin-left: 20px; color: #666;">API: v1</span>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📊 Analytics Summary</h2>
                <div class="metric">
                    <span class="metric-label" data-tooltip="Total number of queries processed by the recursive reasoning system">Total Queries</span>
                    <span class="metric-value skeleton skeleton-number" id="total-queries">–</span>
                </div>
                <div class="metric">
                    <span class="metric-label" data-tooltip="Average improvement in confidence score after refinement (initial → final)">Avg Quality Gain</span>
                    <span class="metric-value skeleton skeleton-number" id="avg-quality-gain">–</span>
                </div>
                <div class="metric">
                    <span class="metric-label" data-tooltip="Average number of refinement iterations per query">Avg Iterations</span>
                    <span class="metric-value skeleton skeleton-number" id="avg-iterations">–</span>
                </div>
                <div class="metric">
                    <span class="metric-label" data-tooltip="Total estimated cost for LLM token usage across all queries">Total Cost</span>
                    <span class="metric-value skeleton skeleton-number" id="total-cost">–</span>
                </div>
            </div>

            <div class="card">
                <h2>🎯 Top Strategies</h2>
                <div id="top-strategies">
                    <div class="skeleton-item" style="width: 80%;"></div>
                    <div class="skeleton-item" style="width: 75%;"></div>
                    <div class="skeleton-item" style="width: 70%;"></div>
                    <div class="skeleton-item" style="width: 65%;"></div>
                </div>
            </div>

            <div class="card">
                <h2>🛠️ Available Skills</h2>
                <div id="skills-list">
                    <div class="skeleton-item" style="width: 85%;"></div>
                    <div class="skeleton-item" style="width: 90%;"></div>
                    <div class="skeleton-item" style="width: 75%;"></div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📝 Recent Executions</h2>
            <table id="executions-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Strategy</th>
                        <th>Query</th>
                        <th>Iterations</th>
                        <th>Quality Gain</th>
                    </tr>
                </thead>
                <tbody id="executions-body">
                    <tr><td colspan="5" style="text-align: center; color: #666;">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let ws;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                console.log('WebSocket connected');
                document.getElementById('ws-status').className = 'connection-status connected';
                document.getElementById('ws-status').textContent = 'Connected';
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                document.getElementById('ws-status').className = 'connection-status disconnected';
                document.getElementById('ws-status').textContent = 'Disconnected';
                setTimeout(connect, 3000); // Reconnect after 3s
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                console.log('Received:', message.type);

                if (message.type === 'initial') {
                    updateDashboard(message.analytics, message.skills, message.recent_executions);
                } else if (message.type === 'analytics_update') {
                    updateAnalytics(message.data);
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                }
            };

            // Ping every 30s
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000);
        }

        function updateDashboard(analytics, skills, executions) {
            updateAnalytics(analytics);
            updateSkills(skills);
            updateExecutions(executions);
        }

        function updateAnalytics(data) {
            // Update metrics and remove skeleton class
            const queriesElement = document.getElementById('total-queries');
            queriesElement.textContent = data.total_queries || 0;
            queriesElement.classList.remove('skeleton');

            const gainElement = document.getElementById('avg-quality-gain');
            gainElement.textContent = ((data.avg_quality_gain || 0) * 100).toFixed(1) + '%';
            gainElement.classList.remove('skeleton');

            const iterElement = document.getElementById('avg-iterations');
            iterElement.textContent = (data.avg_iterations || 0).toFixed(1);
            iterElement.classList.remove('skeleton');

            const costElement = document.getElementById('total-cost');
            costElement.textContent = '$' + (data.total_cost || 0).toFixed(2);
            costElement.classList.remove('skeleton');

            // Update top strategies
            if (data.strategies) {
                const topStrategies = Object.entries(data.strategies)
                    .sort((a, b) => b[1].count - a[1].count)
                    .slice(0, 5);

                const html = topStrategies.map(([name, stats]) =>
                    `<div class="metric">
                        <span class="metric-label">${name}</span>
                        <span class="metric-value">${stats.count} (${(stats.avg_quality_gain * 100).toFixed(1)}%)</span>
                    </div>`
                ).join('');

                document.getElementById('top-strategies').innerHTML = html || 'No data';
            }
        }

        function updateSkills(skills) {
            const categories = Object.entries(skills);
            const html = categories.map(([category, skillList]) =>
                `<div style="margin-bottom: 10px;">
                    <strong style="color: #00d4ff;">${category}</strong>: ${skillList.length}
                </div>`
            ).join('');

            document.getElementById('skills-list').innerHTML = html || 'No skills';
        }

        function updateExecutions(executions) {
            if (!executions || executions.length === 0) {
                document.getElementById('executions-body').innerHTML =
                    '<tr><td colspan="5" style="text-align: center; color: #666;">No executions yet</td></tr>';
                return;
            }

            const rows = executions.map(exec => {
                const time = new Date(exec.timestamp).toLocaleTimeString();
                const qualityGain = ((exec.quality_gain || 0) * 100).toFixed(1) + '%';

                return `<tr>
                    <td class="timestamp">${time}</td>
                    <td>${exec.strategy}</td>
                    <td>${exec.query_text.substring(0, 50)}${exec.query_text.length > 50 ? '...' : ''}</td>
                    <td>${exec.iterations}</td>
                    <td style="color: ${exec.quality_gain > 0 ? '#00ff88' : '#666'}">${qualityGain}</td>
                </tr>`;
            }).join('');

            document.getElementById('executions-body').innerHTML = rows;
        }

        // Connect on page load
        connect();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

# Performance monitoring imports
from HoloLoom.monitoring.performance_metrics import get_metrics_collector
from HoloLoom.monitoring.prometheus_exporter import (
    PrometheusMiddleware,
    metrics_endpoint,
    check_alerts,
    is_prometheus_available
)
from HoloLoom.i18n import TranslationManager

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

# Add Prometheus monitoring middleware
if is_prometheus_available():
    logger.info("Enabling Prometheus metrics collection")
    app.add_middleware(PrometheusMiddleware, update_system_metrics=True)
else:
    logger.warning("Prometheus client not available - metrics collection disabled")

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
i18n: Optional[TranslationManager] = None
metrics_collector = get_metrics_collector()


# ============================================================================
# Language Detection and Management
# ============================================================================

def get_user_locale(request: Request) -> str:
    """
    Detect user's preferred language from multiple sources.

    Detection order:
    1. Query parameter: ?lang=es
    2. Cookie: 'lang' cookie value
    3. Accept-Language header: en-US,en;q=0.9,es;q=0.8 → "en"
    4. Default: "en" (English)

    Args:
        request: FastAPI Request object

    Returns:
        Locale code (e.g., "en", "es", "fr", "de", "zh", "ja")
    """
    # 1. Check query parameter: ?lang=es
    if "lang" in request.query_params:
        requested_lang = request.query_params["lang"]
        if i18n.locale_exists(requested_lang):
            return requested_lang
        logger.debug(f"Unknown locale in query param: {requested_lang}")

    # 2. Check cookie
    if "lang" in request.cookies:
        cookie_lang = request.cookies["lang"]
        if i18n.locale_exists(cookie_lang):
            return cookie_lang
        logger.debug(f"Unknown locale in cookie: {cookie_lang}")

    # 3. Check Accept-Language header
    accept_lang = request.headers.get("accept-language", "en")
    # Parse "en-US,en;q=0.9,es;q=0.8" → preferred languages list
    try:
        preferred_langs = accept_lang.split(',')[0].split('-')[0].lower()
        if i18n.locale_exists(preferred_langs):
            return preferred_langs
    except (IndexError, AttributeError):
        pass

    # 4. Default to English
    return "en"


# ============================================================================
# Initialization
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize dashboard components."""
    global analytics, skill_registry, config, i18n

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

    # Initialize i18n (TranslationManager)
    i18n = TranslationManager(default_locale="en")
    logger.info(f"i18n initialized with {len(i18n.available_locales())} locales: {', '.join(i18n.available_locales())}")

    logger.info("Dashboard server ready! API version: v1, Rate limiting: enabled, i18n: enabled")


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
async def get_dashboard(request: Request):
    """Serve dashboard HTML with user's preferred language."""
    # Detect user locale
    locale = get_user_locale(request)

    # Get translations for this locale
    translations = i18n.get_all(locale)

    # Render dashboard HTML with translations
    html = render_dashboard_with_translations(translations, locale)

    # Create response
    response = HTMLResponse(content=html)

    # Set language cookie to remember user's choice
    response.set_cookie("lang", locale, max_age=31536000)  # 1 year

    return response


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

    @app.get("/api/v1/analytics/charts/quality-trend")
    @limiter.limit("100/minute")
    async def get_quality_trend_chart(request: Request, hours: int = 24):
        """Get quality improvement trend data (hourly buckets)."""
        data = analytics.get_quality_trend(hours=hours)
        return {
            "labels": [d['timestamp'] for d in data],
            "values": [d['avg_quality_gain'] * 100 for d in data],
            "counts": [d['count'] for d in data],
            "min_values": [d['min_gain'] * 100 for d in data],
            "max_values": [d['max_gain'] * 100 for d in data]
        }

    @app.get("/api/v1/analytics/charts/strategy-performance")
    @limiter.limit("100/minute")
    async def get_strategy_performance_chart(request: Request):
        """Get strategy performance comparison data."""
        data = analytics.get_strategy_performance()
        return {
            "labels": [d['strategy'] for d in data],
            "values": [d['avg_quality_gain'] * 100 for d in data],
            "counts": [d['count'] for d in data],
            "colors": [
                '#00ff88' if d['avg_quality_gain'] > 0.15 else
                '#ffaa00' if d['avg_quality_gain'] > 0.10 else
                '#ff4400'
                for d in data
            ]
        }

    @app.get("/api/v1/analytics/charts/query-volume-trend")
    @limiter.limit("100/minute")
    async def get_query_volume_chart(request: Request, days: int = 7):
        """Get query volume trend data (daily buckets)."""
        data = analytics.get_query_volume_trend(days=days)
        return {
            "labels": [d['date'] for d in data],
            "values": [d['count'] for d in data],
            "quality_gains": [d['avg_quality_gain'] * 100 for d in data],
            "durations": [d['avg_duration_ms'] for d in data]
        }

    @app.get("/api/v1/analytics/charts/iteration-distribution")
    @limiter.limit("100/minute")
    async def get_iteration_distribution_chart(request: Request):
        """Get iteration count distribution data."""
        data = analytics.get_iteration_distribution()
        return {
            "labels": list(data.keys()),
            "values": list(data.values()),
            "colors": ['#00d4ff', '#00ff88', '#ffaa00', '#ff4400'][:len(data)]
        }

    @app.get("/api/v1/analytics/charts/cost-analysis")
    @limiter.limit("100/minute")
    async def get_cost_analysis_chart(request: Request):
        """Get cost breakdown by strategy."""
        data = analytics.get_cost_by_strategy()
        return {
            "labels": [d['strategy'] for d in data],
            "input_costs": [d['input_cost'] for d in data],
            "output_costs": [d['output_cost'] for d in data],
            "total_costs": [d['total_cost'] for d in data],
            "counts": [d['count'] for d in data]
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

    @app.get("/api/v1/analytics/charts/quality-trend")
    async def get_quality_trend_chart(request: Request, hours: int = 24):
        """Get quality improvement trend data (hourly buckets)."""
        await limiter(request)
        data = analytics.get_quality_trend(hours=hours)
        return {
            "labels": [d['timestamp'] for d in data],
            "values": [d['avg_quality_gain'] * 100 for d in data],
            "counts": [d['count'] for d in data],
            "min_values": [d['min_gain'] * 100 for d in data],
            "max_values": [d['max_gain'] * 100 for d in data]
        }

    @app.get("/api/v1/analytics/charts/strategy-performance")
    async def get_strategy_performance_chart(request: Request):
        """Get strategy performance comparison data."""
        await limiter(request)
        data = analytics.get_strategy_performance()
        return {
            "labels": [d['strategy'] for d in data],
            "values": [d['avg_quality_gain'] * 100 for d in data],
            "counts": [d['count'] for d in data],
            "colors": [
                '#00ff88' if d['avg_quality_gain'] > 0.15 else
                '#ffaa00' if d['avg_quality_gain'] > 0.10 else
                '#ff4400'
                for d in data
            ]
        }

    @app.get("/api/v1/analytics/charts/query-volume-trend")
    async def get_query_volume_chart(request: Request, days: int = 7):
        """Get query volume trend data (daily buckets)."""
        await limiter(request)
        data = analytics.get_query_volume_trend(days=days)
        return {
            "labels": [d['date'] for d in data],
            "values": [d['count'] for d in data],
            "quality_gains": [d['avg_quality_gain'] * 100 for d in data],
            "durations": [d['avg_duration_ms'] for d in data]
        }

    @app.get("/api/v1/analytics/charts/iteration-distribution")
    async def get_iteration_distribution_chart(request: Request):
        """Get iteration count distribution data."""
        await limiter(request)
        data = analytics.get_iteration_distribution()
        return {
            "labels": list(data.keys()),
            "values": list(data.values()),
            "colors": ['#00d4ff', '#00ff88', '#ffaa00', '#ff4400'][:len(data)]
        }

    @app.get("/api/v1/analytics/charts/cost-analysis")
    async def get_cost_analysis_chart(request: Request):
        """Get cost breakdown by strategy."""
        await limiter(request)
        data = analytics.get_cost_by_strategy()
        return {
            "labels": [d['strategy'] for d in data],
            "input_costs": [d['input_cost'] for d in data],
            "output_costs": [d['output_cost'] for d in data],
            "total_costs": [d['total_cost'] for d in data],
            "counts": [d['count'] for d in data]
        }


# ============================================================================
# Performance Monitoring API Endpoints
# ============================================================================

@app.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics for scraping
    """
    return await metrics_endpoint()


@app.get("/performance")
async def get_performance_dashboard():
    """
    Performance monitoring dashboard page.

    Returns:
        HTML dashboard with real-time performance metrics
    """
    dashboard_path = Path(__file__).parent / "monitoring" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    else:
        return HTMLResponse(
            content="<h1>Performance Dashboard Not Found</h1>"
                    "<p>Dashboard file is missing.</p>",
            status_code=404
        )


@app.get("/api/v1/monitoring/current")
async def get_current_metrics(request: Request):
    """
    Get current performance metrics snapshot.

    Returns:
        JSON with current system, request, performance, and cache metrics
    """
    if RATE_LIMITING_AVAILABLE:
        await limiter(request)

    # Update WebSocket connection count
    metrics_collector.set_active_connections(len(manager.active_connections))

    # Get snapshot
    snapshot = metrics_collector.get_snapshot()

    # Calculate request rate and error rate
    request_rate = metrics_collector.calculate_rate(window_seconds=60)

    # Simple error rate calculation (would be better with proper tracking)
    # For now, return 0 as we don't have error tracking in the snapshot
    error_rate = 0.0

    return {
        "system": {
            "cpu_percent": snapshot.cpu_percent,
            "memory_mb": snapshot.memory_mb,
            "memory_percent": snapshot.memory_percent,
            "active_connections": snapshot.active_connections
        },
        "requests": {
            "total": 0,  # Prometheus counters don't expose values easily
            "rate_per_second": request_rate,
            "error_rate": error_rate
        },
        "performance": {
            "p50_latency_ms": snapshot.p50_latency_ms,
            "p95_latency_ms": snapshot.p95_latency_ms,
            "p99_latency_ms": snapshot.p99_latency_ms
        },
        "cache": {
            "query_cache_hit_rate": snapshot.cache_hit_rates.get('query', 0.0),
            "embedding_cache_hit_rate": snapshot.cache_hit_rates.get('embedding', 0.0)
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/monitoring/history")
async def get_metrics_history(
    request: Request,
    window: str = "1h",
    metric: str = "latency"
):
    """
    Get historical metrics data for charting.

    Args:
        window: Time window (1h, 24h, 7d)
        metric: Metric type (latency, throughput, errors, cache)

    Returns:
        Time-series data for the requested metric
    """
    if RATE_LIMITING_AVAILABLE:
        await limiter(request)

    # This is a placeholder - in production, use a time-series database
    # For now, return empty data
    return {
        "metric": metric,
        "window": window,
        "data": [],
        "message": "Historical data requires time-series database (implement with InfluxDB or Prometheus)"
    }


@app.get("/api/v1/monitoring/alerts")
async def get_active_alerts(request: Request):
    """
    Get active performance alerts.

    Returns:
        List of active alerts with severity, metric, value, threshold
    """
    if RATE_LIMITING_AVAILABLE:
        await limiter(request)

    alerts = await check_alerts()
    return alerts


# ============================================================================
# Anomaly Detection API Endpoints (2025-11-16)
# ============================================================================

# Try to import anomaly detection
try:
    from HoloLoom.monitoring.anomaly_detection import get_anomaly_store
    from HoloLoom.monitoring.performance_metrics import check_metric_anomalies
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False


@app.get("/anomalies")
async def anomaly_dashboard():
    """Serve anomaly detection dashboard."""
    if not ANOMALY_DETECTION_AVAILABLE:
        return HTMLResponse(
            content="<h1>Anomaly Detection Not Available</h1>"
                    "<p>Install dependencies: pip install scikit-learn numpy</p>",
            status_code=503
        )

    # Serve anomaly dashboard HTML
    dashboard_path = Path(__file__).parent / "monitoring" / "anomaly_dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    else:
        # Return embedded HTML
        return HTMLResponse(content=get_anomaly_dashboard_html())


@app.get("/api/v1/monitoring/anomalies/recent")
async def get_recent_anomalies(
    request: Request,
    limit: int = 50,
    severity: Optional[str] = None
):
    """
    Get recent anomalies.

    Args:
        limit: Maximum number of anomalies (default: 50, max: 200)
        severity: Filter by severity (low/medium/high/critical)

    Returns:
        List of recent anomalies with metadata
    """
    if RATE_LIMITING_AVAILABLE:
        await limiter(request)

    if not ANOMALY_DETECTION_AVAILABLE:
        return {"error": "Anomaly detection not available", "anomalies": [], "count": 0}

    # Validate parameters
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")

    if severity and severity not in ['low', 'medium', 'high', 'critical']:
        raise HTTPException(
            status_code=400,
            detail="severity must be one of: low, medium, high, critical"
        )

    store = get_anomaly_store()
    anomalies = store.get_recent(limit=limit, severity=severity)

    return {
        "anomalies": [a.to_dict() for a in anomalies],
        "count": len(anomalies),
        "severity_filter": severity
    }


@app.get("/api/v1/monitoring/anomalies/by-metric/{metric_name}")
async def get_anomalies_by_metric(
    request: Request,
    metric_name: str,
    hours: int = 24
):
    """
    Get anomaly history for a specific metric.

    Args:
        metric_name: Name of the metric
        hours: Time window in hours (default: 24, max: 168)

    Returns:
        Anomalies for the metric in the time window
    """
    if RATE_LIMITING_AVAILABLE:
        await limiter(request)

    if not ANOMALY_DETECTION_AVAILABLE:
        return {"error": "Anomaly detection not available", "anomalies": [], "count": 0}

    # Validate parameters
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 168")

    store = get_anomaly_store()
    anomalies = store.get_by_metric(metric_name, hours=hours)

    return {
        "metric": metric_name,
        "anomalies": [a.to_dict() for a in anomalies],
        "count": len(anomalies),
        "window_hours": hours
    }


@app.post("/api/v1/monitoring/anomalies/{anomaly_id}/acknowledge")
async def acknowledge_anomaly(
    request: Request,
    anomaly_id: int,
    user: str = "system"
):
    """
    Acknowledge an anomaly (mark as reviewed).

    Args:
        anomaly_id: ID of the anomaly
        user: Username of the acknowledger (default: "system")

    Returns:
        Success status
    """
    if RATE_LIMITING_AVAILABLE:
        await limiter(request)

    if not ANOMALY_DETECTION_AVAILABLE:
        return {"error": "Anomaly detection not available"}

    store = get_anomaly_store()
    store.acknowledge(anomaly_id, user)

    return {
        "status": "acknowledged",
        "id": anomaly_id,
        "acknowledged_by": user,
        "acknowledged_at": datetime.now().isoformat()
    }


@app.get("/api/v1/monitoring/anomalies/stats")
async def get_anomaly_stats(
    request: Request,
    hours: int = 24
):
    """
    Get anomaly statistics for dashboard.

    Args:
        hours: Time window in hours (default: 24, max: 168)

    Returns:
        Aggregated anomaly statistics
    """
    if RATE_LIMITING_AVAILABLE:
        await limiter(request)

    if not ANOMALY_DETECTION_AVAILABLE:
        return {
            "error": "Anomaly detection not available",
            "total_anomalies": 0,
            "by_severity": {},
            "by_metric": {},
            "detection_rate": 0.0
        }

    # Validate parameters
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 168")

    store = get_anomaly_store()

    return {
        "total_anomalies": store.count_recent(hours=hours),
        "by_severity": store.count_by_severity(hours=hours),
        "by_metric": store.count_by_metric(hours=hours),
        "detection_rate": store.get_detection_rate(hours=hours),
        "window_hours": hours
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


async def anomaly_detection_loop():
    """Background task to detect anomalies every minute."""
    if not ANOMALY_DETECTION_AVAILABLE:
        logger.warning("Anomaly detection not available - loop disabled")
        return

    logger.info("Starting anomaly detection background loop (60s interval)")

    while True:
        try:
            await asyncio.sleep(60)  # Check every minute

            # Get current metrics snapshot
            snapshot = metrics_collector.get_snapshot()

            # Check for anomalies
            anomalies = check_metric_anomalies(snapshot)

            # Broadcast to connected clients if anomalies detected
            if anomalies and manager.active_connections:
                for anomaly in anomalies:
                    # Only broadcast medium/high/critical
                    if anomaly.severity.value in ['medium', 'high', 'critical']:
                        update = {
                            "type": "anomaly_detected",
                            "anomaly": anomaly.to_dict(),
                            "timestamp": datetime.now().isoformat()
                        }
                        await manager.broadcast(update)

                        logger.warning(
                            f"Anomaly detected: {anomaly.metric_name}={anomaly.metric_value:.2f} "
                            f"(severity: {anomaly.severity.value}, score: {anomaly.anomaly_score:.2f}σ)"
                        )

        except Exception as e:
            logger.error(f"Error in anomaly detection loop: {e}", exc_info=True)
            await asyncio.sleep(60)  # Continue loop even on error


# ============================================================================
# i18n API Endpoints (Language Management)
# ============================================================================

@app.get("/api/v1/locales/available")
async def get_available_locales(request: Request):
    """
    Get list of available locales with metadata.

    Returns:
        List of locale objects with code, name, native name, and flag emoji
    """
    if RATE_LIMITING_AVAILABLE:
        await limiter(request)

    locales = []
    for locale_code in i18n.available_locales():
        meta = i18n.get_locale_metadata(locale_code)
        locales.append(meta)

    return {
        "locales": locales,
        "default": i18n.default_locale,
        "total": len(locales)
    }


@app.get("/api/v1/locales/current")
async def get_current_locale(request: Request):
    """Get the current user's locale."""
    if RATE_LIMITING_AVAILABLE:
        await limiter(request)

    locale = get_user_locale(request)
    meta = i18n.get_locale_metadata(locale)

    return {
        "current": locale,
        "metadata": meta
    }


@app.post("/api/v1/locales/set/{locale}")
async def set_user_locale(locale: str, request: Request):
    """
    Set user's preferred locale.

    Args:
        locale: Locale code (e.g., "es", "fr", "de", "zh", "ja")

    Returns:
        Success status with locale metadata
    """
    if RATE_LIMITING_AVAILABLE:
        await limiter(request)

    # Validate locale
    if not i18n.locale_exists(locale):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown locale: {locale}. Available: {', '.join(i18n.available_locales())}"
        )

    meta = i18n.get_locale_metadata(locale)

    response_data = {
        "success": True,
        "locale": locale,
        "metadata": meta,
        "message": f"Language changed to {meta['name']}"
    }

    response = JSONResponse(content=response_data)

    # Set cookie to remember language preference
    response.set_cookie("lang", locale, max_age=31536000)  # 1 year

    return response


@app.on_event("startup")
async def start_background_tasks():
    """Start background update tasks."""
    asyncio.create_task(broadcast_analytics_updates())
    asyncio.create_task(anomaly_detection_loop())


# ============================================================================
# Dashboard HTML Rendering with i18n
# ============================================================================

def render_dashboard_with_translations(translations: Dict[str, Any], locale: str) -> str:
    """
    Render dashboard HTML with translations for given locale.

    Args:
        translations: Dictionary of translations for the locale
        locale: Locale code (e.g., "en", "es", "fr", "de", "zh", "ja")

    Returns:
        HTML string with all UI elements translated
    """
    # Extract translation keys for easier access
    t = translations

    # Generate language selector HTML
    language_selector_options = ""
    for available_locale in i18n.available_locales():
        meta = i18n.get_locale_metadata(available_locale)
        selected = "selected" if available_locale == locale else ""
        language_selector_options += f'    <option value="{meta["code"]}" {selected}>{meta["flag"]} {meta["name"]}</option>\n'

    # Generate translations JSON for JavaScript
    translations_json = json.dumps(translations)

    return f"""<!DOCTYPE html>
<html lang="{locale}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{t.get('app', {}).get('title', 'HoloLoom Promptly Dashboard')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #2a3555;
        }}
        h1 {{
            color: #00d4ff;
            font-size: 28px;
        }}
        .language-selector {{
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #2a3555;
            background: #1a1f3a;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 14px;
        }}
        .language-selector:hover {{
            border-color: #00d4ff;
            background: #252d4a;
        }}
        .status {{
            background: #1a1f3a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #00d4ff;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background: #1a1f3a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #2a3555;
        }}
        .card h2 {{
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #2a3555;
        }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ color: #888; cursor: help; border-bottom: 1px dotted #555; }}
        .metric-value {{
            color: #00ff88;
            font-weight: 600;
        }}
        .connection-status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .connected {{
            background: #00ff8820;
            color: #00ff88;
        }}
        .disconnected {{
            background: #ff440020;
            color: #ff4400;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        thead {{
            position: sticky;
            top: 0;
            background: #1a1f3a;
            z-index: 10;
        }}
        th, td {{
            padding: 12px 10px;
            text-align: left;
            border-bottom: 1px solid #2a3555;
        }}
        th {{
            color: #00d4ff;
            font-weight: 600;
        }}
        tbody tr {{
            background: #0a0e27;
            transition: background-color 0.2s ease;
        }}
        tbody tr:nth-child(even) {{
            background: #151833;
        }}
        tbody tr:hover {{
            background: #2a3555;
            cursor: pointer;
        }}
        .skeleton {{
            background: linear-gradient(90deg, #1a1f3a 25%, #2a3555 50%, #1a1f3a 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 4px;
            height: 20px;
            margin: 5px 0;
        }}
        .skeleton-item {{
            height: 16px;
            margin-bottom: 10px;
            border-radius: 4px;
            background: linear-gradient(90deg, #1a1f3a 25%, #2a3555 50%, #1a1f3a 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        }}
        @keyframes shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-bar">
            <div>
                <h1>🔮 {t.get('app', {}).get('title', 'HoloLoom Promptly Dashboard')}</h1>
                <p style="color: #888; margin-top: 5px;">{t.get('app', {}).get('subtitle', 'Real-time visualization')}</p>
            </div>
            <select class="language-selector" id="lang-select" onchange="changeLanguage(this.value)">
{language_selector_options}            </select>
        </div>

        <div class="status">
            <strong>{t.get('websocket', {}).get('connection_status', 'WebSocket Connection')}:</strong>
            <span id="ws-status" class="connection-status disconnected">{t.get('websocket', {}).get('disconnected', 'Disconnected')}</span>
            <span style="margin-left: 20px; color: #666;">{t.get('websocket', {}).get('last_update', 'Last update')}: <span id="last-update">–</span></span>
            <span style="margin-left: 20px; color: #666;">API: v1</span>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📊 {t.get('sections', {}).get('analytics_summary', 'Analytics Summary')}</h2>
                <div class="metric">
                    <span class="metric-label" title="{t.get('tooltips', {}).get('total_queries', '')}">{t.get('metrics', {}).get('total_queries', 'Total Queries')}</span>
                    <span class="metric-value skeleton" id="total-queries">–</span>
                </div>
                <div class="metric">
                    <span class="metric-label" title="{t.get('tooltips', {}).get('avg_quality_gain', '')}">{t.get('metrics', {}).get('avg_quality_gain', 'Avg Quality Gain')}</span>
                    <span class="metric-value skeleton" id="avg-quality-gain">–</span>
                </div>
                <div class="metric">
                    <span class="metric-label" title="{t.get('tooltips', {}).get('avg_iterations', '')}">{t.get('metrics', {}).get('avg_iterations', 'Avg Iterations')}</span>
                    <span class="metric-value skeleton" id="avg-iterations">–</span>
                </div>
                <div class="metric">
                    <span class="metric-label" title="{t.get('tooltips', {}).get('total_cost', '')}">{t.get('metrics', {}).get('total_cost', 'Total Cost')}</span>
                    <span class="metric-value skeleton" id="total-cost">–</span>
                </div>
            </div>

            <div class="card">
                <h2>🎯 {t.get('sections', {}).get('top_strategies', 'Top Strategies')}</h2>
                <div id="top-strategies">
                    <div class="skeleton-item" style="width: 80%;"></div>
                    <div class="skeleton-item" style="width: 75%;"></div>
                    <div class="skeleton-item" style="width: 70%;"></div>
                    <div class="skeleton-item" style="width: 65%;"></div>
                </div>
            </div>

            <div class="card">
                <h2>🛠️ {t.get('sections', {}).get('available_skills', 'Available Skills')}</h2>
                <div id="skills-list">
                    <div class="skeleton-item" style="width: 85%;"></div>
                    <div class="skeleton-item" style="width: 90%;"></div>
                    <div class="skeleton-item" style="width: 75%;"></div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📝 {t.get('sections', {}).get('recent_executions', 'Recent Executions')}</h2>
            <table id="executions-table">
                <thead>
                    <tr>
                        <th>{t.get('table', {}).get('time', 'Time')}</th>
                        <th>{t.get('table', {}).get('strategy', 'Strategy')}</th>
                        <th>{t.get('table', {}).get('query', 'Query')}</th>
                        <th>{t.get('table', {}).get('iterations', 'Iterations')}</th>
                        <th>{t.get('table', {}).get('quality_gain', 'Quality Gain')}</th>
                    </tr>
                </thead>
                <tbody id="executions-body">
                    <tr><td colspan="5" style="text-align: center; color: #666;">{t.get('empty_states', {}).get('no_executions', 'No executions yet')}</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Translations object for JavaScript
        const T = {translations_json};
        const currentLocale = "{locale}";

        let ws;

        function changeLanguage(newLang) {{
            // Call API to set language
            fetch(`/api/v1/locales/set/${{newLang}}`, {{ method: 'POST' }})
                .then(() => {{
                    // Reload page with new language
                    window.location.href = `/?lang=${{newLang}}`;
                }})
                .catch(err => console.error('Error changing language:', err));
        }}

        function connect() {{
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${{protocol}}//${{window.location.host}}/ws`);

            ws.onopen = () => {{
                console.log('WebSocket connected');
                document.getElementById('ws-status').className = 'connection-status connected';
                document.getElementById('ws-status').textContent = T.websocket.connected;
            }};

            ws.onclose = () => {{
                console.log('WebSocket disconnected');
                document.getElementById('ws-status').className = 'connection-status disconnected';
                document.getElementById('ws-status').textContent = T.websocket.disconnected;
                setTimeout(connect, 3000);
            }};

            ws.onmessage = (event) => {{
                const message = JSON.parse(event.data);
                console.log('Received:', message.type);

                if (message.type === 'initial') {{
                    updateDashboard(message.analytics, message.skills, message.recent_executions);
                }} else if (message.type === 'analytics_update') {{
                    updateAnalytics(message.data);
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                }}
            }};

            // Ping every 30s
            setInterval(() => {{
                if (ws.readyState === WebSocket.OPEN) {{
                    ws.send(JSON.stringify({{type: 'ping'}}));
                }}
            }}, 30000);
        }}

        function updateDashboard(analytics, skills, executions) {{
            updateAnalytics(analytics);
            updateSkills(skills);
            updateExecutions(executions);
        }}

        function updateAnalytics(data) {{
            const queriesElement = document.getElementById('total-queries');
            queriesElement.textContent = data.total_queries || 0;
            queriesElement.classList.remove('skeleton');

            const gainElement = document.getElementById('avg-quality-gain');
            gainElement.textContent = (((data.avg_quality_gain || 0) * 100).toFixed(1)) + T.units.percent;
            gainElement.classList.remove('skeleton');

            const iterElement = document.getElementById('avg-iterations');
            iterElement.textContent = (data.avg_iterations || 0).toFixed(1);
            iterElement.classList.remove('skeleton');

            const costElement = document.getElementById('total-cost');
            costElement.textContent = T.units.dollars + (data.total_cost || 0).toFixed(2);
            costElement.classList.remove('skeleton');

            if (data.strategies) {{
                const topStrategies = Object.entries(data.strategies)
                    .sort((a, b) => b[1].count - a[1].count)
                    .slice(0, 5);

                const html = topStrategies.map(([name, stats]) =>
                    `<div class="metric">
                        <span class="metric-label">${{name}}</span>
                        <span class="metric-value">${{stats.count}} (${{(stats.avg_quality_gain * 100).toFixed(1)}}%)</span>
                    </div>`
                ).join('');

                document.getElementById('top-strategies').innerHTML = html || T.empty_states.no_strategies;
            }}
        }}

        function updateSkills(skills) {{
            const categories = Object.entries(skills);
            const html = categories.map(([category, skillList]) =>
                `<div style="margin-bottom: 10px;">
                    <strong style="color: #00d4ff;">${{category}}</strong>: ${{skillList.length}}
                </div>`
            ).join('');

            document.getElementById('skills-list').innerHTML = html || T.empty_states.no_skills;
        }}

        function updateExecutions(executions) {{
            if (!executions || executions.length === 0) {{
                document.getElementById('executions-body').innerHTML =
                    `<tr><td colspan="5" style="text-align: center; color: #666;">${{T.empty_states.no_executions}}</td></tr>`;
                return;
            }}

            const rows = executions.map(exec => {{
                const time = new Date(exec.timestamp).toLocaleTimeString();
                const qualityGain = ((exec.quality_gain || 0) * 100).toFixed(1) + T.units.percent;

                return `<tr>
                    <td class="timestamp">${{time}}</td>
                    <td>${{exec.strategy}}</td>
                    <td>${{exec.query_text.substring(0, 50)}}${{exec.query_text.length > 50 ? '...' : ''}}</td>
                    <td>${{exec.iterations}}</td>
                    <td style="color: ${{exec.quality_gain > 0 ? '#00ff88' : '#666'}}">${{qualityGain}}</td>
                </tr>`;
            }}).join('');

            document.getElementById('executions-body').innerHTML = rows;
        }}

        // Connect on page load
        connect();
    </script>
</body>
</html>
"""


# ============================================================================
# Embedded Dashboard HTML (fallback)
# ============================================================================

def get_embedded_dashboard_html() -> str:
    """Get embedded dashboard HTML with API v1 endpoints and Chart.js visualizations."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>HoloLoom Promptly Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
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
        .chart-card {
            background: #1a1f3a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #2a3555;
            margin-bottom: 20px;
            min-height: 400px;
            display: flex;
            flex-direction: column;
        }
        .chart-card h2 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 18px;
        }
        canvas {
            max-height: 300px;
            margin-top: auto;
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
        tbody tr {
            background: #0a0e27;
            transition: background-color 0.2s ease;
        }
        tbody tr:nth-child(even) {
            background: #151833;
        }
        tbody tr:hover {
            background: #2a3555;
            cursor: pointer;
        }
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
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        @media (max-width: 1200px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
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
                    <span class="metric-label" data-tooltip="Total number of queries processed">Total Queries</span>
                    <span class="metric-value skeleton skeleton-number" id="total-queries">–</span>
                </div>
                <div class="metric">
                    <span class="metric-label" data-tooltip="Average quality improvement after refinement">Avg Quality Gain</span>
                    <span class="metric-value skeleton skeleton-number" id="avg-quality-gain">–</span>
                </div>
                <div class="metric">
                    <span class="metric-label" data-tooltip="Average refinement iterations per query">Avg Iterations</span>
                    <span class="metric-value skeleton skeleton-number" id="avg-iterations">–</span>
                </div>
                <div class="metric">
                    <span class="metric-label" data-tooltip="Total LLM token cost">Total Cost</span>
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

        <h2 style="color: #00d4ff; margin-top: 30px; margin-bottom: 20px;">📈 Analytics Charts</h2>

        <div class="charts-grid">
            <div class="chart-card">
                <h2>📈 Quality Improvement Trend (24h)</h2>
                <canvas id="qualityTrendChart"></canvas>
            </div>

            <div class="chart-card">
                <h2>📊 Strategy Performance</h2>
                <canvas id="strategyPerformanceChart"></canvas>
            </div>

            <div class="chart-card">
                <h2>📅 Query Volume Trend (7 days)</h2>
                <canvas id="queryVolumeTrendChart"></canvas>
            </div>

            <div class="chart-card">
                <h2>🔄 Iteration Distribution</h2>
                <canvas id="iterationDistributionChart"></canvas>
            </div>

            <div class="chart-card" style="grid-column: span 2;">
                <h2>💰 Cost Analysis by Strategy</h2>
                <canvas id="costAnalysisChart"></canvas>
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
        // Chart.js default configuration for dark theme
        Chart.defaults.color = '#e0e0e0';
        Chart.defaults.borderColor = '#2a3555';

        let ws;
        let charts = {};

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                console.log('WebSocket connected');
                document.getElementById('ws-status').className = 'connection-status connected';
                document.getElementById('ws-status').textContent = 'Connected';
                updateCharts();
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                document.getElementById('ws-status').className = 'connection-status disconnected';
                document.getElementById('ws-status').textContent = 'Disconnected';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                if (message.type === 'initial') {
                    updateDashboard(message.analytics, message.skills, message.recent_executions);
                    updateCharts();
                } else if (message.type === 'analytics_update') {
                    updateAnalytics(message.data);
                    updateCharts();
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                }
            };

            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000);
        }

        function initializeCharts() {
            // Quality Trend Chart (Line)
            const qualityCtx = document.getElementById('qualityTrendChart').getContext('2d');
            charts.quality = new Chart(qualityCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Avg Quality Gain (%)',
                        data: [],
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: true, labels: { color: '#00d4ff' } } },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: '#00d4ff' },
                            grid: { color: '#2a3555' }
                        },
                        x: {
                            ticks: { color: '#888' },
                            grid: { display: false }
                        }
                    }
                }
            });

            // Strategy Performance Chart (Bar)
            const strategyCtx = document.getElementById('strategyPerformanceChart').getContext('2d');
            charts.strategy = new Chart(strategyCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Avg Quality Gain (%)',
                        data: [],
                        backgroundColor: []
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: '#00d4ff' },
                            grid: { color: '#2a3555' }
                        },
                        x: {
                            ticks: { color: '#888' },
                            grid: { display: false }
                        }
                    }
                }
            });

            // Query Volume Trend Chart (Area)
            const volumeCtx = document.getElementById('queryVolumeTrendChart').getContext('2d');
            charts.volume = new Chart(volumeCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Query Count',
                        data: [],
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.2)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: true, labels: { color: '#00d4ff' } } },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: '#00d4ff' },
                            grid: { color: '#2a3555' }
                        },
                        x: {
                            ticks: { color: '#888' },
                            grid: { display: false }
                        }
                    }
                }
            });

            // Iteration Distribution Chart (Doughnut)
            const iterCtx = document.getElementById('iterationDistributionChart').getContext('2d');
            charts.iteration = new Chart(iterCtx, {
                type: 'doughnut',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: ['#00d4ff', '#00ff88', '#ffaa00', '#ff4400']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: true, labels: { color: '#e0e0e0' } } }
                }
            });

            // Cost Analysis Chart (Stacked Bar)
            const costCtx = document.getElementById('costAnalysisChart').getContext('2d');
            charts.cost = new Chart(costCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Input Cost ($)',
                            data: [],
                            backgroundColor: '#00d4ff'
                        },
                        {
                            label: 'Output Cost ($)',
                            data: [],
                            backgroundColor: '#00ff88'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: { legend: { display: true, labels: { color: '#e0e0e0' } } },
                    scales: {
                        x: {
                            stacked: true,
                            ticks: { color: '#00d4ff' },
                            grid: { color: '#2a3555' }
                        },
                        y: {
                            stacked: true,
                            ticks: { color: '#888' },
                            grid: { display: false }
                        }
                    }
                }
            });
        }

        async function updateCharts() {
            try {
                // Quality Trend
                const qualityData = await fetch('/api/v1/analytics/charts/quality-trend').then(r => r.json());
                if (charts.quality) {
                    charts.quality.data.labels = qualityData.labels;
                    charts.quality.data.datasets[0].data = qualityData.values;
                    charts.quality.update();
                }

                // Strategy Performance
                const strategyData = await fetch('/api/v1/analytics/charts/strategy-performance').then(r => r.json());
                if (charts.strategy) {
                    charts.strategy.data.labels = strategyData.labels;
                    charts.strategy.data.datasets[0].data = strategyData.values;
                    charts.strategy.data.datasets[0].backgroundColor = strategyData.colors;
                    charts.strategy.update();
                }

                // Query Volume
                const volumeData = await fetch('/api/v1/analytics/charts/query-volume-trend').then(r => r.json());
                if (charts.volume) {
                    charts.volume.data.labels = volumeData.labels;
                    charts.volume.data.datasets[0].data = volumeData.values;
                    charts.volume.update();
                }

                // Iteration Distribution
                const iterData = await fetch('/api/v1/analytics/charts/iteration-distribution').then(r => r.json());
                if (charts.iteration) {
                    charts.iteration.data.labels = iterData.labels;
                    charts.iteration.data.datasets[0].data = iterData.values;
                    charts.iteration.update();
                }

                // Cost Analysis
                const costData = await fetch('/api/v1/analytics/charts/cost-analysis').then(r => r.json());
                if (charts.cost) {
                    charts.cost.data.labels = costData.labels;
                    charts.cost.data.datasets[0].data = costData.input_costs;
                    charts.cost.data.datasets[1].data = costData.output_costs;
                    charts.cost.update();
                }
            } catch (error) {
                console.warn('Chart update failed:', error);
            }
        }

        function updateDashboard(analytics, skills, executions) {
            updateAnalytics(analytics);
            updateSkills(skills);
            updateExecutions(executions);
        }

        function updateAnalytics(data) {
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

        // Initialize on page load
        window.addEventListener('DOMContentLoaded', () => {
            initializeCharts();
            connect();
            setInterval(updateCharts, 5000);
        });
    </script>
</body>
</html>
"""


def get_anomaly_dashboard_html() -> str:
    """Get embedded anomaly detection dashboard HTML."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>HoloLoom Anomaly Detection</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { color: #00d4ff; margin-bottom: 10px; font-size: 28px; }
        .header-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #2a3555;
        }
        .filters {
            background: #1a1f3a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
        }
        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .filter-group label {
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
        }
        select, input[type="number"] {
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #2a3555;
            background: #0a0e27;
            color: #e0e0e0;
            cursor: pointer;
        }
        select:hover, input:hover { border-color: #00d4ff; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #1a1f3a;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #2a3555;
            text-align: center;
        }
        .stat-value {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .stat-value.critical { color: #ff4400; }
        .stat-value.high { color: #ffaa00; }
        .stat-value.medium { color: #00d4ff; }
        .stat-value.low { color: #00ff88; }
        .stat-label { color: #888; font-size: 12px; text-transform: uppercase; }
        .timeline-section {
            background: #1a1f3a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #2a3555;
            margin-bottom: 20px;
        }
        .timeline-section h2 { color: #00d4ff; margin-bottom: 15px; font-size: 18px; }
        .feed-section {
            background: #1a1f3a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #2a3555;
        }
        .feed-section h2 { color: #00d4ff; margin-bottom: 15px; font-size: 18px; }
        .anomaly-item {
            background: #0a0e27;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 10px;
            border-left: 4px solid;
            position: relative;
        }
        .anomaly-item.critical { border-left-color: #ff4400; }
        .anomaly-item.high { border-left-color: #ffaa00; }
        .anomaly-item.medium { border-left-color: #00d4ff; }
        .anomaly-item.low { border-left-color: #00ff88; }
        .anomaly-item.acknowledged {
            opacity: 0.5;
            background: #151833;
        }
        .anomaly-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 10px;
        }
        .anomaly-severity {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .anomaly-severity.critical { background: #ff440020; color: #ff4400; }
        .anomaly-severity.high { background: #ffaa0020; color: #ffaa00; }
        .anomaly-severity.medium { background: #00d4ff20; color: #00d4ff; }
        .anomaly-severity.low { background: #00ff8820; color: #00ff88; }
        .anomaly-metric {
            font-size: 16px;
            font-weight: 600;
            color: #e0e0e0;
        }
        .anomaly-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .detail-item {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .detail-label { color: #888; font-size: 11px; }
        .detail-value { color: #e0e0e0; font-weight: 500; }
        .anomaly-actions {
            margin-top: 10px;
            display: flex;
            gap: 10px;
        }
        .btn {
            padding: 6px 12px;
            border-radius: 4px;
            border: 1px solid #2a3555;
            background: #1a1f3a;
            color: #00d4ff;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        .btn:hover {
            background: #2a3555;
            border-color: #00d4ff;
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .timestamp {
            color: #666;
            font-size: 11px;
        }
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .connection-status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .connected { background: #00ff8820; color: #00ff88; }
        .disconnected { background: #ff440020; color: #ff4400; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-bar">
            <div>
                <h1>🚨 Anomaly Detection Dashboard</h1>
                <p style="color: #888; margin-top: 5px;">Real-time monitoring with ML-based detection</p>
            </div>
            <div>
                <span style="color: #666; margin-right: 15px;">WebSocket: <span id="ws-status" class="connection-status disconnected">Disconnected</span></span>
                <a href="/" style="color: #00d4ff; text-decoration: none;">← Back to Main Dashboard</a>
            </div>
        </div>

        <div class="filters">
            <div class="filter-group">
                <label>Severity</label>
                <select id="severity-filter" onchange="applyFilters()">
                    <option value="">All Severities</option>
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Metric</label>
                <select id="metric-filter" onchange="applyFilters()">
                    <option value="">All Metrics</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Time Window (hours)</label>
                <input type="number" id="hours-filter" value="24" min="1" max="168" onchange="applyFilters()">
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="total-anomalies">0</div>
                <div class="stat-label">Total Anomalies</div>
            </div>
            <div class="stat-card">
                <div class="stat-value critical" id="critical-count">0</div>
                <div class="stat-label">Critical</div>
            </div>
            <div class="stat-card">
                <div class="stat-value high" id="high-count">0</div>
                <div class="stat-label">High</div>
            </div>
            <div class="stat-card">
                <div class="stat-value medium" id="medium-count">0</div>
                <div class="stat-label">Medium</div>
            </div>
            <div class="stat-card">
                <div class="stat-value low" id="low-count">0</div>
                <div class="stat-label">Low</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="detection-rate">0.0</div>
                <div class="stat-label">Per Hour</div>
            </div>
        </div>

        <div class="timeline-section">
            <h2>📈 Anomaly Timeline</h2>
            <canvas id="anomalyTimeline" style="max-height: 250px;"></canvas>
        </div>

        <div class="feed-section">
            <h2>🔔 Recent Anomalies</h2>
            <div id="anomaly-feed">
                <div class="empty-state">Loading anomalies...</div>
            </div>
        </div>
    </div>

    <script>
        Chart.defaults.color = '#e0e0e0';
        Chart.defaults.borderColor = '#2a3555';

        let ws;
        let allAnomalies = [];
        let timelineChart;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                console.log('WebSocket connected');
                document.getElementById('ws-status').className = 'connection-status connected';
                document.getElementById('ws-status').textContent = 'Connected';
                loadAnomalies();
                loadStats();
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                document.getElementById('ws-status').className = 'connection-status disconnected';
                document.getElementById('ws-status').textContent = 'Disconnected';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                if (message.type === 'anomaly_detected') {
                    handleNewAnomaly(message.anomaly);
                }
            };
        }

        async function loadAnomalies() {
            try {
                const severity = document.getElementById('severity-filter').value;
                const url = severity
                    ? `/api/v1/monitoring/anomalies/recent?limit=100&severity=${severity}`
                    : '/api/v1/monitoring/anomalies/recent?limit=100';

                const response = await fetch(url);
                const data = await response.json();

                allAnomalies = data.anomalies || [];
                renderAnomalies(allAnomalies);
                updateMetricFilter(allAnomalies);
                updateTimeline(allAnomalies);
            } catch (error) {
                console.error('Error loading anomalies:', error);
            }
        }

        async function loadStats() {
            try {
                const hours = document.getElementById('hours-filter').value;
                const response = await fetch(`/api/v1/monitoring/anomalies/stats?hours=${hours}`);
                const stats = await response.json();

                document.getElementById('total-anomalies').textContent = stats.total_anomalies || 0;
                document.getElementById('critical-count').textContent = stats.by_severity?.critical || 0;
                document.getElementById('high-count').textContent = stats.by_severity?.high || 0;
                document.getElementById('medium-count').textContent = stats.by_severity?.medium || 0;
                document.getElementById('low-count').textContent = stats.by_severity?.low || 0;
                document.getElementById('detection-rate').textContent = (stats.detection_rate || 0).toFixed(1);
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }

        function renderAnomalies(anomalies) {
            const feed = document.getElementById('anomaly-feed');

            if (anomalies.length === 0) {
                feed.innerHTML = '<div class="empty-state">No anomalies detected in this time window.</div>';
                return;
            }

            feed.innerHTML = anomalies.map(anomaly => {
                const time = new Date(anomaly.timestamp).toLocaleString();
                const acknowledgedClass = anomaly.acknowledged ? 'acknowledged' : '';

                return `
                    <div class="anomaly-item ${anomaly.severity} ${acknowledgedClass}" data-id="${anomaly.id}">
                        <div class="anomaly-header">
                            <div>
                                <span class="anomaly-severity ${anomaly.severity}">${anomaly.severity}</span>
                                <div class="anomaly-metric">${anomaly.metric_name}</div>
                            </div>
                            <div class="timestamp">${time}</div>
                        </div>

                        <div class="anomaly-details">
                            <div class="detail-item">
                                <div class="detail-label">Current Value</div>
                                <div class="detail-value">${anomaly.metric_value.toFixed(2)}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Expected Range</div>
                                <div class="detail-value">
                                    ${anomaly.expected_min ? anomaly.expected_min.toFixed(2) : 'N/A'} -
                                    ${anomaly.expected_max ? anomaly.expected_max.toFixed(2) : 'N/A'}
                                </div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Anomaly Score</div>
                                <div class="detail-value">${anomaly.anomaly_score.toFixed(2)}σ</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Detection Method</div>
                                <div class="detail-value">${anomaly.detection_method}</div>
                            </div>
                        </div>

                        <div class="anomaly-actions">
                            ${anomaly.acknowledged
                                ? `<span style="color: #00ff88; font-size: 12px;">✓ Acknowledged by ${anomaly.acknowledged_by}</span>`
                                : `<button class="btn" onclick="acknowledgeAnomaly(${anomaly.id})">Acknowledge</button>`
                            }
                            <button class="btn" onclick="viewDetails(${anomaly.id})">View Details</button>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function updateMetricFilter(anomalies) {
            const metrics = new Set(anomalies.map(a => a.metric_name));
            const filter = document.getElementById('metric-filter');
            const currentValue = filter.value;

            filter.innerHTML = '<option value="">All Metrics</option>';
            metrics.forEach(metric => {
                const option = document.createElement('option');
                option.value = metric;
                option.textContent = metric;
                filter.appendChild(option);
            });

            filter.value = currentValue;
        }

        function updateTimeline(anomalies) {
            // Group by hour
            const hourlyData = {};
            anomalies.forEach(anomaly => {
                const hour = new Date(anomaly.timestamp).toISOString().slice(0, 13) + ':00:00';
                if (!hourlyData[hour]) {
                    hourlyData[hour] = { critical: 0, high: 0, medium: 0, low: 0 };
                }
                hourlyData[hour][anomaly.severity]++;
            });

            const labels = Object.keys(hourlyData).sort();
            const criticalData = labels.map(l => hourlyData[l].critical);
            const highData = labels.map(l => hourlyData[l].high);
            const mediumData = labels.map(l => hourlyData[l].medium);
            const lowData = labels.map(l => hourlyData[l].low);

            if (timelineChart) {
                timelineChart.destroy();
            }

            const ctx = document.getElementById('anomalyTimeline').getContext('2d');
            timelineChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels.map(l => new Date(l).toLocaleString()),
                    datasets: [
                        {
                            label: 'Critical',
                            data: criticalData,
                            backgroundColor: '#ff4400',
                        },
                        {
                            label: 'High',
                            data: highData,
                            backgroundColor: '#ffaa00',
                        },
                        {
                            label: 'Medium',
                            data: mediumData,
                            backgroundColor: '#00d4ff',
                        },
                        {
                            label: 'Low',
                            data: lowData,
                            backgroundColor: '#00ff88',
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { stacked: true, ticks: { color: '#888' }, grid: { display: false } },
                        y: { stacked: true, ticks: { color: '#00d4ff' }, grid: { color: '#2a3555' } }
                    },
                    plugins: { legend: { display: true, labels: { color: '#e0e0e0' } } }
                }
            });
        }

        function handleNewAnomaly(anomaly) {
            allAnomalies.unshift(anomaly);
            renderAnomalies(allAnomalies);
            loadStats();
            updateTimeline(allAnomalies);
        }

        async function acknowledgeAnomaly(id) {
            try {
                const response = await fetch(`/api/v1/monitoring/anomalies/${id}/acknowledge`, {
                    method: 'POST'
                });

                if (response.ok) {
                    loadAnomalies();
                }
            } catch (error) {
                console.error('Error acknowledging anomaly:', error);
            }
        }

        function viewDetails(id) {
            const anomaly = allAnomalies.find(a => a.id === id);
            if (anomaly) {
                alert(JSON.stringify(anomaly, null, 2));
            }
        }

        function applyFilters() {
            loadAnomalies();
            loadStats();
        }

        // Initialize
        connect();
        setInterval(() => {
            loadAnomalies();
            loadStats();
        }, 30000); // Refresh every 30 seconds
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

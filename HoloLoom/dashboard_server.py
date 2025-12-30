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
- REST API for historical data
- FastAPI backend
- Connects to analytics database
- Live monitoring of all Promptly integration features

Usage:
    uvicorn HoloLoom.dashboard_server:app --reload --port 8000

Then open: http://localhost:8000

Created: 2025-11-16
Integration: Phases 1-4 → Real-time Dashboard
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# HoloLoom imports
from HoloLoom.config import Config
from HoloLoom.telemetry.analytics.recursive_analytics import RecursiveAnalytics
from HoloLoom.weaving_orchestrator_recursive import RecursiveWeavingOrchestrator
from HoloLoom.agentic.skill_agents import SkillRegistry, list_available_skills

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="HoloLoom Promptly Dashboard", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    logger.info("Dashboard server ready!")


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
# WebSocket Endpoint
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
# REST API Endpoints
# ============================================================================

@app.get("/")
async def get_dashboard():
    """Serve dashboard HTML."""
    dashboard_html = Path(__file__).parent / "dashboard.html"

    if dashboard_html.exists():
        return FileResponse(dashboard_html)
    else:
        return HTMLResponse(content=get_embedded_dashboard_html())


@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary."""
    summary = analytics.get_summary()
    return summary


@app.get("/api/analytics/trends")
async def get_analytics_trends(days: int = 7):
    """Get quality trends over time."""
    trends = analytics.get_quality_trends(days=days)
    return {"trends": trends}


@app.get("/api/analytics/strategy/{strategy}")
async def get_strategy_metrics(strategy: str):
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


@app.get("/api/analytics/recommendations")
async def get_recommendations():
    """Get AI-powered recommendations."""
    recommendations = analytics.get_recommendations()
    return {"recommendations": recommendations}


@app.get("/api/skills")
async def get_skills():
    """Get all available skills."""
    skills = await list_available_skills()
    return skills


@app.get("/api/skills/{skill_name}")
async def get_skill_details(skill_name: str):
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


@app.get("/api/executions/recent")
async def get_recent_executions(limit: int = 20):
    """Get recent executions."""
    recent = analytics.get_recent_executions(limit=limit)
    return {"executions": recent}


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
    """Get embedded dashboard HTML."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>HoloLoom Promptly Dashboard</title>
    <meta charset="utf-8">
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
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #2a3555;
        }
        th {
            color: #00d4ff;
            font-weight: 600;
        }
        .timestamp {
            color: #666;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 HoloLoom Promptly Dashboard</h1>
        <div class="status">
            <strong>WebSocket:</strong> <span id="ws-status" class="connection-status disconnected">Disconnected</span>
            <span style="margin-left: 20px; color: #666;">Last update: <span id="last-update">Never</span></span>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📊 Analytics Summary</h2>
                <div class="metric">
                    <span class="metric-label">Total Queries</span>
                    <span class="metric-value" id="total-queries">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Quality Gain</span>
                    <span class="metric-value" id="avg-quality-gain">0%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Iterations</span>
                    <span class="metric-value" id="avg-iterations">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Cost</span>
                    <span class="metric-value" id="total-cost">$0.00</span>
                </div>
            </div>

            <div class="card">
                <h2>🎯 Top Strategies</h2>
                <div id="top-strategies">Loading...</div>
            </div>

            <div class="card">
                <h2>🛠️ Available Skills</h2>
                <div id="skills-list">Loading...</div>
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
            document.getElementById('total-queries').textContent = data.total_queries || 0;
            document.getElementById('avg-quality-gain').textContent =
                ((data.avg_quality_gain || 0) * 100).toFixed(1) + '%';
            document.getElementById('avg-iterations').textContent =
                (data.avg_iterations || 0).toFixed(1);
            document.getElementById('total-cost').textContent =
                '$' + (data.total_cost || 0).toFixed(2);

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

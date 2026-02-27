"""
Monitor Router
==============

Agent monitoring, sessions, metrics, and WebSocket endpoints.
Extracted from agentic_api.py as part of W2 (Monolithic Files) SWOT remediation.

Endpoints:
- GET /api/monitor/sessions - Get all active agent sessions
- GET /api/monitor/sessions/{agent_id} - Get specific session
- GET /api/monitor/projects - Get all projects
- GET /api/monitor/projects/{project} - Get project agents
- GET /api/monitor/metrics - Get performance metrics
- WS /ws/monitor - WebSocket for real-time monitoring
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring"])


# ============================================================================
# Check if monitoring is available
# ============================================================================

try:
    from hololoom.agentic.monitoring import get_monitor, start_monitoring, stop_monitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    get_monitor = None
    start_monitoring = None
    stop_monitoring = None


# ============================================================================
# Dependencies
# ============================================================================

def get_server_state(request: Request):
    """
    Dependency to get server state from the app.

    The state is attached to the app during startup in agentic_api.py.
    """
    return request.app.state.server_state


def require_monitoring():
    """
    Dependency to check if monitoring is available.

    Raises HTTPException if monitoring is not available.
    """
    if not MONITORING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Monitoring not available. Install monitoring dependencies."
        )
    return True


# ============================================================================
# REST Endpoints
# ============================================================================

@router.get("/api/monitor/sessions")
async def get_monitor_sessions(
    state=Depends(get_server_state),
    _=Depends(require_monitoring)
):
    """
    Get all active agent sessions.

    Returns:
        List of active session summaries with agent IDs and metadata.

    Example:
        GET /api/monitor/sessions

        Response:
        {
          "sessions": [
            {
              "agent_id": "agent-123",
              "project": "my-project",
              "started_at": "2025-11-15T10:30:00Z",
              "queries_processed": 42,
              "avg_latency_ms": 125.5
            },
            ...
          ],
          "total": 5
        }
    """
    try:
        monitor = get_monitor()
        if not monitor:
            return {"sessions": [], "total": 0}

        sessions = monitor.get_all_sessions()

        return {
            "sessions": [
                {
                    "agent_id": s.agent_id,
                    "project": s.project,
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                    "queries_processed": s.queries_processed,
                    "avg_latency_ms": s.avg_latency_ms
                }
                for s in sessions
            ],
            "total": len(sessions)
        }

    except Exception as e:
        logger.error(f"Failed to get monitor sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/monitor/sessions/{agent_id}")
async def get_monitor_session(
    agent_id: str,
    state=Depends(get_server_state),
    _=Depends(require_monitoring)
):
    """
    Get details for a specific agent session.

    Args:
        agent_id: The agent identifier

    Returns:
        Detailed session information including query history

    Example:
        GET /api/monitor/sessions/agent-123

        Response:
        {
          "agent_id": "agent-123",
          "project": "my-project",
          "started_at": "2025-11-15T10:30:00Z",
          "queries_processed": 42,
          "avg_latency_ms": 125.5,
          "recent_queries": [...],
          "memory_usage_mb": 256.5
        }
    """
    try:
        monitor = get_monitor()
        if not monitor:
            raise HTTPException(status_code=404, detail="Monitor not initialized")

        session = monitor.get_session(agent_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {agent_id}")

        return {
            "agent_id": session.agent_id,
            "project": session.project,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "queries_processed": session.queries_processed,
            "avg_latency_ms": session.avg_latency_ms,
            "recent_queries": session.recent_queries[-10:] if hasattr(session, 'recent_queries') else [],
            "memory_usage_mb": session.memory_usage_mb if hasattr(session, 'memory_usage_mb') else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/monitor/projects")
async def get_monitor_projects(
    state=Depends(get_server_state),
    _=Depends(require_monitoring)
):
    """
    Get all monitored projects.

    Returns:
        List of projects with agent counts

    Example:
        GET /api/monitor/projects

        Response:
        {
          "projects": [
            {
              "name": "my-project",
              "agent_count": 3,
              "total_queries": 150,
              "avg_latency_ms": 98.2
            },
            ...
          ],
          "total": 2
        }
    """
    try:
        monitor = get_monitor()
        if not monitor:
            return {"projects": [], "total": 0}

        projects = monitor.get_all_projects()

        return {
            "projects": [
                {
                    "name": p.name,
                    "agent_count": p.agent_count,
                    "total_queries": p.total_queries,
                    "avg_latency_ms": p.avg_latency_ms
                }
                for p in projects
            ],
            "total": len(projects)
        }

    except Exception as e:
        logger.error(f"Failed to get projects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/monitor/projects/{project}")
async def get_monitor_project_agents(
    project: str,
    state=Depends(get_server_state),
    _=Depends(require_monitoring)
):
    """
    Get all agents for a specific project.

    Args:
        project: The project name

    Returns:
        List of agents in the project

    Example:
        GET /api/monitor/projects/my-project

        Response:
        {
          "project": "my-project",
          "agents": [
            {"agent_id": "agent-123", "queries_processed": 42},
            {"agent_id": "agent-456", "queries_processed": 28}
          ],
          "total": 2
        }
    """
    try:
        monitor = get_monitor()
        if not monitor:
            raise HTTPException(status_code=404, detail="Monitor not initialized")

        agents = monitor.get_project_agents(project)
        if agents is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project}")

        return {
            "project": project,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "queries_processed": a.queries_processed,
                    "avg_latency_ms": a.avg_latency_ms if hasattr(a, 'avg_latency_ms') else None
                }
                for a in agents
            ],
            "total": len(agents)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agents for project {project}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/monitor/metrics")
async def get_monitor_metrics(
    state=Depends(get_server_state),
    _=Depends(require_monitoring)
):
    """
    Get aggregated performance metrics.

    Returns:
        System-wide performance metrics

    Example:
        GET /api/monitor/metrics

        Response:
        {
          "total_queries": 1523,
          "total_sessions": 15,
          "total_projects": 5,
          "avg_latency_ms": 112.5,
          "p95_latency_ms": 245.0,
          "cache_hit_rate": 0.73,
          "success_rate": 0.98
        }
    """
    try:
        monitor = get_monitor()
        if not monitor:
            return {
                "total_queries": 0,
                "total_sessions": 0,
                "total_projects": 0,
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "cache_hit_rate": 0,
                "success_rate": 0
            }

        metrics = monitor.get_metrics()

        return {
            "total_queries": metrics.total_queries,
            "total_sessions": metrics.total_sessions,
            "total_projects": metrics.total_projects,
            "avg_latency_ms": metrics.avg_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "cache_hit_rate": metrics.cache_hit_rate,
            "success_rate": metrics.success_rate
        }

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@router.websocket("/ws/monitor")
async def websocket_monitor(
    websocket: WebSocket,
    project: Optional[str] = None
):
    """
    WebSocket endpoint for real-time monitoring updates.

    Sends periodic updates about agent sessions and metrics.
    Can be filtered by project.

    Query Parameters:
        project: Optional project name to filter updates

    Message Format (Server → Client):
        {
          "type": "update",
          "timestamp": "2025-11-15T10:30:00Z",
          "data": {
            "sessions": [...],
            "metrics": {...}
          }
        }

    Client Commands:
        {"type": "subscribe", "project": "my-project"}
        {"type": "unsubscribe"}
        {"type": "ping"}

    Example:
        ws://localhost:8000/ws/monitor?project=my-project
    """
    if not MONITORING_AVAILABLE:
        await websocket.close(code=4001, reason="Monitoring not available")
        return

    await websocket.accept()
    logger.info(f"WebSocket monitor connected (project={project})")

    try:
        monitor = get_monitor()
        if not monitor:
            await websocket.close(code=4002, reason="Monitor not initialized")
            return

        # Subscribe to updates
        subscription = None
        if project:
            subscription = project

        while True:
            try:
                # Wait for client message or timeout for periodic update
                import asyncio
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=5.0  # Send update every 5 seconds
                    )

                    # Handle client commands
                    msg_type = message.get("type")
                    if msg_type == "subscribe":
                        subscription = message.get("project")
                        await websocket.send_json({
                            "type": "subscribed",
                            "project": subscription
                        })
                    elif msg_type == "unsubscribe":
                        subscription = None
                        await websocket.send_json({
                            "type": "unsubscribed"
                        })
                    elif msg_type == "ping":
                        await websocket.send_json({
                            "type": "pong"
                        })

                except asyncio.TimeoutError:
                    # Send periodic update
                    pass

                # Get current state
                from datetime import datetime

                if subscription:
                    sessions = [
                        s for s in monitor.get_all_sessions()
                        if s.project == subscription
                    ]
                else:
                    sessions = monitor.get_all_sessions()

                metrics = monitor.get_metrics()

                await websocket.send_json({
                    "type": "update",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "sessions": [
                            {
                                "agent_id": s.agent_id,
                                "project": s.project,
                                "queries_processed": s.queries_processed
                            }
                            for s in sessions
                        ],
                        "metrics": {
                            "total_queries": metrics.total_queries,
                            "avg_latency_ms": metrics.avg_latency_ms,
                            "success_rate": metrics.success_rate
                        }
                    }
                })

            except WebSocketDisconnect:
                logger.info(f"WebSocket monitor disconnected (project={project})")
                break

    except Exception as e:
        logger.error(f"WebSocket monitor error: {e}", exc_info=True)
        await websocket.close(code=4003, reason=str(e))

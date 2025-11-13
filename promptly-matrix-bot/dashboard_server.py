#!/usr/bin/env python3
"""
Dashboard WebSocket Server for Promptly Matrix Bot

Provides real-time updates for:
- Weaving cycle visualization
- Knowledge graph updates
- Audit trail events
- Team collaboration changes

Usage:
    python dashboard_server.py

WebSocket endpoint: ws://localhost:8000/ws
REST API: http://localhost:8000/api/*
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn

# Import bot components
from bot.hololoom_integration_enhanced import EnhancedHoloLoomBot as HoloLoomBot, WeavingResponse
from bot.audit_trail import AuditTrail, EventType, Outcome
from bot.team_context import TeamContext, ContextScope, Permission

# GitHub integration (Phase 5)
try:
    from bot.github_integration import GitHubIntegration, GitHubWebhookHandler
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    logger.warning("GitHub integration not available")

# Production hardening (Phase 6)
from bot.metrics import metrics, track_request, track_hololoom_query
from bot.error_recovery import with_retry, BackoffStrategy
from bot.circuit_breaker import global_registry as circuit_breaker_registry
from bot.rate_limiter import global_user_limiter, global_room_limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Promptly Dashboard API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
bot: Optional[HoloLoomBot] = None
audit_trail: Optional[AuditTrail] = None
team_context: Optional[TeamContext] = None
github: Optional['GitHubIntegration'] = None
github_webhook: Optional['GitHubWebhookHandler'] = None
active_connections: Set[WebSocket] = set()
query_history: List[Dict] = []


class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """Accept and track connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove connection"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message)

        # Send to all, removing dead connections
        dead_connections = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                dead_connections.add(connection)

        # Clean up dead connections
        self.active_connections -= dead_connections


manager = ConnectionManager()


@dataclass
class WeavingStepUpdate:
    """Real-time weaving step update"""
    step: int
    name: str
    status: str  # waiting, in_progress, completed, error
    latency_ms: Optional[float] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None


async def broadcast_weaving_step(
    query_id: str,
    query_text: str,
    step: int,
    name: str,
    status: str,
    latency_ms: Optional[float] = None,
    metadata: Optional[Dict] = None
):
    """Broadcast weaving step update to all clients"""
    await manager.broadcast({
        "type": "weaving_update",
        "data": {
            "query_id": query_id,
            "query_text": query_text,
            "step": step,
            "name": name,
            "status": status,
            "latency_ms": latency_ms,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
    })


# Startup/Shutdown Events
@app.on_event("startup")
async def startup():
    """Initialize bot and services"""
    global bot, audit_trail, team_context, github, github_webhook

    logger.info("Starting Promptly Dashboard Server...")

    # Initialize HoloLoom bot with production features
    try:
        # Create WebSocket callback for real-time weaving updates
        async def weaving_callback(data: Dict):
            """Broadcast HoloLoom weaving updates to all connected clients"""
            await manager.broadcast(data)

        # Initialize Enhanced HoloLoom bot
        bot = HoloLoomBot(
            config_mode="FAST",
            enable_production_hardening=True,
            enable_neo4j=False,  # Use NetworkX (set to True for production Neo4j)
            websocket_callback=weaving_callback
        )

        # Use async context manager initialization
        await bot.initialize()
        logger.info("Enhanced HoloLoom bot initialized with production hardening")
        logger.info(f"  - Dynamic Yarn Graph (KG) memory backend")
        logger.info(f"  - Real-time WebSocket updates enabled")
        logger.info(f"  - Circuit breakers and rate limiting active")
    except Exception as e:
        logger.error(f"Failed to initialize HoloLoom bot: {e}")
        bot = None

    # Initialize audit trail
    audit_trail = AuditTrail()
    logger.info("Audit trail initialized")

    # Initialize team context
    team_context = TeamContext()
    logger.info("Team context initialized")

    # Initialize GitHub integration (Phase 5)
    if GITHUB_AVAILABLE:
        try:
            import os
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                github = GitHubIntegration(token=github_token)
                github_webhook = GitHubWebhookHandler(secret=os.getenv("GITHUB_WEBHOOK_SECRET"))
                logger.info("GitHub integration initialized")
            else:
                logger.warning("GITHUB_TOKEN not set. GitHub integration disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize GitHub integration: {e}")

    logger.info("Dashboard server ready!")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup resources"""
    global bot

    logger.info("Shutting down dashboard server...")

    if bot:
        try:
            await bot.cleanup()
            logger.info("HoloLoom bot cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during bot cleanup: {e}")

    logger.info("Shutdown complete")


# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "data": {"message": "Connected to Promptly Dashboard"},
            "timestamp": datetime.now().isoformat()
        })

        # Keep connection alive
        while True:
            # Receive messages from client (ping/pong, commands, etc.)
            data = await websocket.receive_text()

            # Handle ping
            if data == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# REST API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "bot_initialized": bot is not None and bot.initialized,
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics"""
    if not bot:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    stats = await bot.get_statistics()

    # Add query history stats
    total_queries = len(query_history)
    avg_latency = sum(q.get('latency_ms', 0) for q in query_history) / max(total_queries, 1)
    avg_confidence = sum(q.get('confidence', 0) for q in query_history) / max(total_queries, 1)

    return {
        "bot_stats": stats,
        "query_stats": {
            "total_queries": total_queries,
            "avg_latency_ms": avg_latency,
            "avg_confidence": avg_confidence,
            "recent_queries": query_history[-10:]  # Last 10 queries
        },
        "active_connections": len(manager.active_connections)
    }


@app.post("/api/query")
async def process_query(query: Dict):
    """
    Process a query through HoloLoom weaving cycle with real-time updates

    Body:
        {
            "text": "What is Thompson Sampling?",
            "user_id": "@alice:matrix.org",
            "room_id": "!room:matrix.org",
            "complexity": "FAST"
        }
    """
    if not bot:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    query_text = query.get("text", "")
    user_id = query.get("user_id", "@unknown:matrix.org")
    room_id = query.get("room_id")
    complexity = query.get("complexity", "FAST")

    if not query_text:
        raise HTTPException(status_code=400, detail="Query text required")

    # Generate query ID
    import hashlib
    query_id = hashlib.md5(f"{query_text}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    # Execute weaving - real-time updates handled by bot's websocket_callback
    try:
        response: WeavingResponse = await bot.weave(
            query=query_text,
            user_id=user_id,
            room_id=room_id,
            complexity=complexity
        )

        # Log to audit trail
        if audit_trail:
            await audit_trail.log_command(
                user=user_id,
                room=room_id or "unknown",
                command="weave",
                args={"query": query_text, "complexity": complexity},
                outcome=Outcome.SUCCESS,
                metadata={
                    "confidence": response.confidence,
                    "tool_used": response.tool_used,
                    "latency_ms": response.latency_ms
                }
            )

        # Add to query history with enhanced data
        query_record = {
            "query_id": query_id,
            "query_text": query_text,
            "timestamp": datetime.now().isoformat(),
            "confidence": response.confidence,
            "tool_used": response.tool_used,
            "latency_ms": response.latency_ms,
            "cache_hit": response.cache_hit,
            "response": response.text,
            "context_depth": response.context_depth,
            "memories_retrieved": response.memories_retrieved,
            "stages": response.stages  # Real stage data from orchestrator
        }
        query_history.append(query_record)

        # Note: Final "weaving_complete" already broadcast by bot's websocket_callback

        return {
            "success": True,
            "data": {
                "query_id": query_id,
                "response": response.text,
                "confidence": response.confidence,
                "tool_used": response.tool_used,
                "latency_ms": response.latency_ms,
                "cache_hit": response.cache_hit,
                "context_depth": response.context_depth,
                "memories_retrieved": response.memories_retrieved,
                "stages": response.stages,
                "provenance": response.provenance,
                "summary": response.summary()
            }
        }

    except Exception as e:
        logger.error(f"Query processing failed: {e}")

        # Broadcast error
        await manager.broadcast({
            "type": "weaving_error",
            "data": {
                "query_id": query_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        })

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audit")
async def get_audit_trail(
    limit: int = 50,
    event_type: Optional[str] = None,
    user: Optional[str] = None,
    outcome: Optional[str] = None
):
    """Get audit trail events with filtering"""
    if not audit_trail:
        raise HTTPException(status_code=503, detail="Audit trail not initialized")

    # Query events
    events = await audit_trail.query(
        event_type=EventType[event_type.upper()] if event_type else None,
        user=user,
        outcome=Outcome[outcome.upper()] if outcome else None
    )

    # Limit results
    events = events[:limit]

    return {
        "success": True,
        "data": {
            "events": [asdict(e) for e in events],
            "total": len(events)
        }
    }


@app.get("/api/prompts")
async def get_shared_prompts(scope: Optional[str] = None):
    """Get shared prompts library"""
    if not team_context:
        raise HTTPException(status_code=503, detail="Team context not initialized")

    # Get all prompts or filter by scope
    if scope:
        prompts = await team_context.get_prompts_by_scope(ContextScope[scope.upper()])
    else:
        prompts = await team_context.get_all_prompts()

    return {
        "success": True,
        "data": {
            "prompts": [asdict(p) for p in prompts]
        }
    }


@app.post("/api/prompts")
async def create_prompt(prompt_data: Dict):
    """Create a new shared prompt"""
    if not team_context:
        raise HTTPException(status_code=503, detail="Team context not initialized")

    try:
        prompt_id = await team_context.create_prompt(
            title=prompt_data.get("title"),
            content=prompt_data.get("content"),
            author=prompt_data.get("author"),
            scope=ContextScope[prompt_data.get("scope", "USER").upper()],
            tags=prompt_data.get("tags", [])
        )

        return {
            "success": True,
            "data": {
                "prompt_id": prompt_id,
                "message": "Prompt created successfully"
            }
        }
    except Exception as e:
        logger.error(f"Failed to create prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/prompts/{prompt_id}")
async def update_prompt(prompt_id: str, prompt_data: Dict):
    """Update an existing prompt"""
    if not team_context:
        raise HTTPException(status_code=503, detail="Team context not initialized")

    try:
        await team_context.update_prompt(
            prompt_id=prompt_id,
            title=prompt_data.get("title"),
            content=prompt_data.get("content"),
            tags=prompt_data.get("tags")
        )

        return {
            "success": True,
            "data": {
                "message": "Prompt updated successfully"
            }
        }
    except Exception as e:
        logger.error(f"Failed to update prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a prompt"""
    if not team_context:
        raise HTTPException(status_code=503, detail="Team context not initialized")

    try:
        await team_context.delete_prompt(prompt_id)

        return {
            "success": True,
            "data": {
                "message": "Prompt deleted successfully"
            }
        }
    except Exception as e:
        logger.error(f"Failed to delete prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/permissions")
async def get_permissions():
    """Get all user permissions"""
    if not team_context:
        raise HTTPException(status_code=503, detail="Team context not initialized")

    try:
        permissions = await team_context.get_all_permissions()

        return {
            "success": True,
            "data": {
                "permissions": [asdict(p) for p in permissions]
            }
        }
    except Exception as e:
        logger.error(f"Failed to get permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/permissions")
async def grant_permission(permission_data: Dict):
    """Grant permission to a user"""
    if not team_context:
        raise HTTPException(status_code=503, detail="Team context not initialized")

    try:
        await team_context.grant_permission(
            user_id=permission_data.get("user_id"),
            role=Permission[permission_data.get("role", "VIEWER").upper()],
            granted_by=permission_data.get("granted_by", "@system:matrix.org")
        )

        return {
            "success": True,
            "data": {
                "message": "Permission granted successfully"
            }
        }
    except Exception as e:
        logger.error(f"Failed to grant permission: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/permissions/{user_id}")
async def revoke_permission(user_id: str):
    """Revoke permission from a user"""
    if not team_context:
        raise HTTPException(status_code=503, detail="Team context not initialized")

    try:
        await team_context.revoke_permission(user_id)

        return {
            "success": True,
            "data": {
                "message": "Permission revoked successfully"
            }
        }
    except Exception as e:
        logger.error(f"Failed to revoke permission: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/usage")
async def get_usage_analytics():
    """Get usage analytics for team collaboration"""
    if not team_context:
        raise HTTPException(status_code=503, detail="Team context not initialized")

    try:
        # Get all prompts for analytics
        all_prompts = await team_context.get_all_prompts()

        # Calculate analytics
        total_prompts = len(all_prompts)
        total_usage = sum(p.usage_count for p in all_prompts)
        active_users = len(set(p.author for p in all_prompts))

        # Popular prompts (top 5)
        popular_prompts = sorted(
            all_prompts,
            key=lambda p: p.usage_count,
            reverse=True
        )[:5]

        # Usage by scope
        usage_by_scope = {
            "TEAM": sum(1 for p in all_prompts if p.scope == ContextScope.TEAM),
            "ROOM": sum(1 for p in all_prompts if p.scope == ContextScope.ROOM),
            "USER": sum(1 for p in all_prompts if p.scope == ContextScope.USER),
        }

        # Recent activity (last 10 from audit trail)
        recent_activity = []
        if audit_trail:
            events = await audit_trail.query(limit=10)
            recent_activity = [
                {
                    "timestamp": e.timestamp,
                    "user": e.user,
                    "action": e.action,
                    "prompt_title": e.metadata.get("prompt_title", "Unknown")
                }
                for e in events
                if e.event_type == EventType.COMMAND
            ]

        return {
            "success": True,
            "data": {
                "total_prompts": total_prompts,
                "total_usage": total_usage,
                "active_users": active_users,
                "popular_prompts": [
                    {
                        "prompt_id": p.id,
                        "title": p.title,
                        "usage_count": p.usage_count
                    }
                    for p in popular_prompts
                ],
                "usage_by_scope": usage_by_scope,
                "recent_activity": recent_activity
            }
        }
    except Exception as e:
        logger.error(f"Failed to get usage analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph")
async def get_knowledge_graph():
    """Get knowledge graph structure"""
    if not bot or not bot.initialized:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    # Extract graph from bot knowledge shards
    nodes = []
    edges = []
    entity_connections = {}

    # Build graph from bot's knowledge shards
    for shard in bot.shards:
        # Add entities as nodes
        for entity in shard.entities:
            entity_id = entity.lower().replace(" ", "_")
            if entity_id not in entity_connections:
                entity_connections[entity_id] = 0
                nodes.append({
                    "id": entity_id,
                    "label": entity,
                    "type": "entity",
                    "connections": 0,
                    "metadata": {"source_shard": shard.id}
                })

        # Add motifs as nodes
        for motif in shard.motifs:
            motif_id = motif.lower().replace(" ", "_")
            if motif_id not in entity_connections:
                entity_connections[motif_id] = 0
                nodes.append({
                    "id": motif_id,
                    "label": motif,
                    "type": "motif",
                    "connections": 0,
                    "metadata": {"source_shard": shard.id}
                })

        # Create edges between entities in same shard (co-occurrence)
        for i, entity1 in enumerate(shard.entities):
            entity1_id = entity1.lower().replace(" ", "_")
            for entity2 in shard.entities[i+1:]:
                entity2_id = entity2.lower().replace(" ", "_")
                edges.append({
                    "source": entity1_id,
                    "target": entity2_id,
                    "type": "MENTIONS",
                    "weight": 0.8,
                    "metadata": {"shard": shard.id}
                })
                entity_connections[entity1_id] = entity_connections.get(entity1_id, 0) + 1
                entity_connections[entity2_id] = entity_connections.get(entity2_id, 0) + 1

        # Create edges from entities to motifs
        for entity in shard.entities:
            entity_id = entity.lower().replace(" ", "_")
            for motif in shard.motifs:
                motif_id = motif.lower().replace(" ", "_")
                if entity_id != motif_id:
                    edges.append({
                        "source": entity_id,
                        "target": motif_id,
                        "type": "PART_OF",
                        "weight": 1.0,
                        "metadata": {"shard": shard.id}
                    })
                    entity_connections[entity_id] = entity_connections.get(entity_id, 0) + 1
                    entity_connections[motif_id] = entity_connections.get(motif_id, 0) + 1

    # Update connection counts
    for node in nodes:
        node["connections"] = entity_connections.get(node["id"], 0)

    # Add some inferred relationships based on common patterns
    # IS_A relationships (basic taxonomy)
    taxonomy_rules = [
        ("thompson_sampling", "algorithm", "IS_A"),
        ("bayesian", "statistical_method", "IS_A"),
        ("exploration", "strategy", "IS_A"),
    ]

    for source, target, rel_type in taxonomy_rules:
        # Check if both nodes exist
        if source in entity_connections and target in entity_connections:
            edges.append({
                "source": source,
                "target": target,
                "type": rel_type,
                "weight": 1.0,
                "metadata": {"inferred": True}
            })

    return {
        "success": True,
        "data": {
            "nodes": nodes,
            "edges": edges
        }
    }


@app.post("/api/workflow/execute")
async def execute_workflow(workflow_data: Dict):
    """
    Execute a workflow with real-time node status updates

    Body:
        {
            "workflow": {
                "version": "1.0",
                "name": "My Workflow",
                "nodes": [...],
                "connections": [...]
            },
            "input_data": {
                "query": "Sample query"
            }
        }
    """
    if not bot:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    workflow = workflow_data.get("workflow", {})
    input_data = workflow_data.get("input_data", {})

    workflow_id = hashlib.md5(f"{workflow.get('name')}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    logger.info(f"Executing workflow: {workflow.get('name')} (ID: {workflow_id})")

    # Track node results
    node_results = {}
    start_time = datetime.now()

    try:
        # Execute nodes in topological order
        nodes = workflow.get("nodes", [])
        connections = workflow.get("connections", [])

        # Build adjacency list for topological sort
        adjacency = {node["id"]: [] for node in nodes}
        in_degree = {node["id"]: 0 for node in nodes}

        for conn in connections:
            adjacency[conn["from"]].append(conn["to"])
            in_degree[conn["to"]] += 1

        # Topological sort (Kahn's algorithm)
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            current = queue.pop(0)
            execution_order.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Execute nodes in order
        for node_id in execution_order:
            node = next((n for n in nodes if n["id"] == node_id), None)
            if not node:
                continue

            node_type = node["type"]
            logger.info(f"Executing node {node_id} ({node_type})")

            # Simulate node execution (in real implementation, call actual agent)
            await asyncio.sleep(0.1)  # Simulate processing

            # Store result
            node_results[node_id] = {
                "status": "completed",
                "output": f"Output from {node_type}",
                "latency_ms": 100,
                "error": None
            }

        # Calculate total duration
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Log to audit trail
        if audit_trail:
            await audit_trail.log_workflow_execution(
                user="@dashboard:matrix.org",
                workflow_name=workflow.get("name"),
                outcome=Outcome.SUCCESS,
                metadata={
                    "workflow_id": workflow_id,
                    "nodes_executed": len(execution_order),
                    "duration_ms": duration_ms
                }
            )

        return {
            "success": True,
            "data": {
                "workflow_id": workflow_id,
                "status": "completed",
                "node_results": node_results,
                "duration_ms": duration_ms,
                "execution_order": execution_order
            }
        }

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")

        # Log failure to audit trail
        if audit_trail:
            await audit_trail.log_workflow_execution(
                user="@dashboard:matrix.org",
                workflow_name=workflow.get("name"),
                outcome=Outcome.FAILURE,
                metadata={
                    "workflow_id": workflow_id,
                    "error": str(e)
                }
            )

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows")
async def save_workflow(workflow: Dict):
    """
    Save a workflow for future use

    Body:
        {
            "id": "wf_123",
            "name": "My Workflow",
            "description": "Description",
            "nodes": [...],
            "edges": [...],
            "created_at": "2025-11-13T...",
            "updated_at": "2025-11-13T..."
        }
    """
    # In a real implementation, this would save to a database
    # For now, we'll just log it

    workflow_id = workflow.get("id")
    workflow_name = workflow.get("name")

    logger.info(f"Saving workflow: {workflow_name} (ID: {workflow_id})")

    # Log to audit trail
    if audit_trail:
        await audit_trail.log_config_change(
            user="@dashboard:matrix.org",
            change_type="workflow_save",
            before=None,
            after=workflow_name,
            metadata={
                "workflow_id": workflow_id,
                "node_count": len(workflow.get("nodes", [])),
                "edge_count": len(workflow.get("edges", []))
            }
        )

    return {
        "success": True,
        "data": {
            "message": f"Workflow '{workflow_name}' saved successfully",
            "workflow_id": workflow_id
        }
    }


# GitHub Integration Endpoints (Phase 5)

@app.post("/api/github/pr")
async def create_github_pr(pr_data: Dict):
    """
    Create a GitHub pull request from a workflow or dashboard

    Body:
        {
            "repo": "owner/repo",
            "title": "PR Title",
            "body": "PR Description",
            "head": "feature-branch",
            "base": "main",
            "files": {
                "path/to/file.py": "file content"
            },
            "draft": false
        }
    """
    if not GITHUB_AVAILABLE or not github:
        raise HTTPException(status_code=503, detail="GitHub integration not available")

    try:
        result = await github.create_pull_request(
            repo=pr_data.get("repo"),
            title=pr_data.get("title"),
            body=pr_data.get("body"),
            head=pr_data.get("head"),
            base=pr_data.get("base", "main"),
            files=pr_data.get("files"),
            draft=pr_data.get("draft", False)
        )

        # Log to audit trail
        if audit_trail:
            await audit_trail.log_command(
                user=pr_data.get("user", "@dashboard:matrix.org"),
                room=None,
                command="github_pr_create",
                outcome=Outcome.SUCCESS,
                metadata={
                    "repo": pr_data.get("repo"),
                    "pr_number": result.pr_number,
                    "pr_url": result.pr_html_url
                }
            )

        return {
            "success": True,
            "data": {
                "pr_number": result.pr_number,
                "pr_url": result.pr_html_url,
                "created_at": result.created_at,
                "status": result.status.value
            }
        }
    except Exception as e:
        logger.error(f"Failed to create PR: {e}")

        # Log error to audit trail
        if audit_trail:
            await audit_trail.log_command(
                user=pr_data.get("user", "@dashboard:matrix.org"),
                room=None,
                command="github_pr_create",
                outcome=Outcome.ERROR,
                error=str(e),
                metadata={"repo": pr_data.get("repo")}
            )

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/github/pr/{pr_number}/review")
async def review_github_pr(pr_number: int, review_data: Dict):
    """
    Review a GitHub pull request with automated code review

    Body:
        {
            "repo": "owner/repo",
            "auto_approve": false,
            "user": "@reviewer:matrix.org"
        }
    """
    if not GITHUB_AVAILABLE or not github:
        raise HTTPException(status_code=503, detail="GitHub integration not available")

    try:
        result = await github.review_pull_request(
            repo=review_data.get("repo"),
            pr_number=pr_number,
            auto_approve=review_data.get("auto_approve", False)
        )

        # Log to audit trail
        if audit_trail:
            await audit_trail.log_command(
                user=review_data.get("user", "@dashboard:matrix.org"),
                room=None,
                command="github_pr_review",
                outcome=Outcome.SUCCESS if result.approved else Outcome.PARTIAL,
                metadata={
                    "repo": review_data.get("repo"),
                    "pr_number": pr_number,
                    "approved": result.approved,
                    "issues_found": result.issues_found,
                    "security_issues": result.security_issues,
                    "quality_score": result.quality_score
                }
            )

        return {
            "success": True,
            "data": {
                "approved": result.approved,
                "issues_found": result.issues_found,
                "security_issues": result.security_issues,
                "quality_score": result.quality_score,
                "comments": result.comments,
                "summary": result.summary
            }
        }
    except Exception as e:
        logger.error(f"Failed to review PR: {e}")

        # Log error to audit trail
        if audit_trail:
            await audit_trail.log_command(
                user=review_data.get("user", "@dashboard:matrix.org"),
                room=None,
                command="github_pr_review",
                outcome=Outcome.ERROR,
                error=str(e),
                metadata={"repo": review_data.get("repo"), "pr_number": pr_number}
            )

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/github/issue")
async def create_github_issue(issue_data: Dict):
    """
    Create a GitHub issue

    Body:
        {
            "repo": "owner/repo",
            "title": "Issue Title",
            "body": "Issue Description",
            "labels": ["bug", "urgent"],
            "assignee": "username",
            "milestone": 1,
            "user": "@creator:matrix.org"
        }
    """
    if not GITHUB_AVAILABLE or not github:
        raise HTTPException(status_code=503, detail="GitHub integration not available")

    try:
        result = await github.create_issue(
            repo=issue_data.get("repo"),
            title=issue_data.get("title"),
            body=issue_data.get("body"),
            labels=issue_data.get("labels"),
            assignee=issue_data.get("assignee"),
            milestone=issue_data.get("milestone")
        )

        # Log to audit trail
        if audit_trail:
            await audit_trail.log_command(
                user=issue_data.get("user", "@dashboard:matrix.org"),
                room=None,
                command="github_issue_create",
                outcome=Outcome.SUCCESS,
                metadata={
                    "repo": issue_data.get("repo"),
                    "issue_number": result.issue_number,
                    "issue_url": result.issue_html_url
                }
            )

        return {
            "success": True,
            "data": {
                "issue_number": result.issue_number,
                "issue_url": result.issue_html_url,
                "created_at": result.created_at,
                "status": result.status.value
            }
        }
    except Exception as e:
        logger.error(f"Failed to create issue: {e}")

        # Log error to audit trail
        if audit_trail:
            await audit_trail.log_command(
                user=issue_data.get("user", "@dashboard:matrix.org"),
                room=None,
                command="github_issue_create",
                outcome=Outcome.ERROR,
                error=str(e),
                metadata={"repo": issue_data.get("repo")}
            )

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/github/workflow/trigger")
async def trigger_github_workflow(workflow_data: Dict):
    """
    Trigger a GitHub Actions workflow

    Body:
        {
            "repo": "owner/repo",
            "workflow_id": "build.yml",
            "ref": "main",
            "inputs": {
                "version": "1.0.0",
                "environment": "production"
            },
            "user": "@trigger:matrix.org"
        }
    """
    if not GITHUB_AVAILABLE or not github:
        raise HTTPException(status_code=503, detail="GitHub integration not available")

    try:
        success = await github.trigger_workflow(
            repo=workflow_data.get("repo"),
            workflow_id=workflow_data.get("workflow_id"),
            ref=workflow_data.get("ref", "main"),
            inputs=workflow_data.get("inputs")
        )

        # Log to audit trail
        if audit_trail:
            await audit_trail.log_command(
                user=workflow_data.get("user", "@dashboard:matrix.org"),
                room=None,
                command="github_workflow_trigger",
                outcome=Outcome.SUCCESS if success else Outcome.ERROR,
                metadata={
                    "repo": workflow_data.get("repo"),
                    "workflow_id": workflow_data.get("workflow_id"),
                    "ref": workflow_data.get("ref", "main")
                }
            )

        return {
            "success": success,
            "data": {
                "message": f"Workflow {workflow_data.get('workflow_id')} triggered successfully" if success else "Failed to trigger workflow"
            }
        }
    except Exception as e:
        logger.error(f"Failed to trigger workflow: {e}")

        # Log error to audit trail
        if audit_trail:
            await audit_trail.log_command(
                user=workflow_data.get("user", "@dashboard:matrix.org"),
                room=None,
                command="github_workflow_trigger",
                outcome=Outcome.ERROR,
                error=str(e),
                metadata={"repo": workflow_data.get("repo"), "workflow_id": workflow_data.get("workflow_id")}
            )

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/github/webhook")
async def handle_github_webhook(webhook_data: Dict):
    """
    Handle GitHub webhook events

    Headers:
        X-GitHub-Event: Event type (push, pull_request, issue, workflow_run)

    Body:
        GitHub webhook payload (varies by event type)
    """
    if not GITHUB_AVAILABLE or not github_webhook:
        raise HTTPException(status_code=503, detail="GitHub webhook handler not available")

    try:
        event_type = webhook_data.get("event_type", "unknown")
        payload = webhook_data.get("payload", {})

        result = await github_webhook.handle_event(event_type, payload)

        # Log to audit trail
        if audit_trail:
            await audit_trail.log_event(
                event_type=EventType.SYSTEM,
                user="@github:webhook",
                room=None,
                action="webhook_received",
                outcome=Outcome.SUCCESS,
                metadata={
                    "event_type": event_type,
                    "status": result.get("status"),
                    "repo": payload.get("repository", {}).get("full_name")
                }
            )

        # Broadcast webhook event to dashboard clients
        await manager.broadcast({
            "type": "github_webhook",
            "data": {
                "event_type": event_type,
                "status": result.get("status"),
                "timestamp": datetime.now().isoformat()
            }
        })

        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"Failed to handle webhook: {e}")

        # Log error to audit trail
        if audit_trail:
            await audit_trail.log_event(
                event_type=EventType.SYSTEM,
                user="@github:webhook",
                room=None,
                action="webhook_received",
                outcome=Outcome.ERROR,
                error=str(e),
                metadata={"event_type": webhook_data.get("event_type", "unknown")}
            )

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/github/pr/{pr_number}/status")
async def get_github_pr_status(pr_number: int, repo: str):
    """
    Get GitHub PR status (checks, reviews, mergeable)

    Query params:
        repo: Repository name (owner/repo)
    """
    if not GITHUB_AVAILABLE or not github:
        raise HTTPException(status_code=503, detail="GitHub integration not available")

    try:
        status = await github.get_pr_status(repo=repo, pr_number=pr_number)

        return {
            "success": True,
            "data": status
        }
    except Exception as e:
        logger.error(f"Failed to get PR status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Production Hardening Endpoints (Phase 6)

@app.get("/metrics")
async def get_prometheus_metrics():
    """
    Export Prometheus metrics

    Returns metrics in Prometheus text format for scraping
    """
    return Response(
        content=metrics.export(),
        media_type="text/plain; version=0.0.4"
    )


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint with circuit breaker status

    Returns:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "timestamp": ISO timestamp,
            "circuit_breakers": {...},
            "rate_limiters": {...}
        }
    """
    # Get circuit breaker status
    cb_health = circuit_breaker_registry.get_health_status()

    # Get rate limiter stats
    user_limiter_stats = global_user_limiter.get_stats()
    room_limiter_stats = global_room_limiter.get_stats()

    # Determine overall health
    if cb_health["healthy"] and user_limiter_stats.denial_rate() < 0.5:
        status = "healthy"
    elif cb_health["open_breakers"] > 0 or user_limiter_stats.denial_rate() > 0.8:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "circuit_breakers": cb_health,
        "rate_limiters": {
            "user_limiter": {
                "total_requests": user_limiter_stats.total_requests,
                "allowed": user_limiter_stats.allowed_requests,
                "denied": user_limiter_stats.denied_requests,
                "denial_rate": user_limiter_stats.denial_rate(),
                "requests_per_second": user_limiter_stats.requests_per_second()
            },
            "room_limiter": {
                "total_requests": room_limiter_stats.total_requests,
                "allowed": room_limiter_stats.allowed_requests,
                "denied": room_limiter_stats.denied_requests,
                "denial_rate": room_limiter_stats.denial_rate(),
                "requests_per_second": room_limiter_stats.requests_per_second()
            }
        }
    }


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "dashboard_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

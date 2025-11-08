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
from fastapi.responses import JSONResponse
import uvicorn

# Import bot components
from bot.hololoom_integration import HoloLoomBot, WeavingResponse
from bot.audit_trail import AuditTrail, EventType, Outcome
from bot.team_context import TeamContext, ContextScope, Permission

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
    global bot, audit_trail, team_context

    logger.info("Starting Promptly Dashboard Server...")

    # Initialize HoloLoom bot
    try:
        bot = HoloLoomBot(config_mode="FAST")
        await bot.initialize()
        logger.info("HoloLoom bot initialized")
    except Exception as e:
        logger.error(f"Failed to initialize HoloLoom bot: {e}")
        bot = None

    # Initialize audit trail
    audit_trail = AuditTrail()
    logger.info("Audit trail initialized")

    # Initialize team context
    team_context = TeamContext()
    logger.info("Team context initialized")

    logger.info("Dashboard server ready!")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup resources"""
    global bot

    logger.info("Shutting down dashboard server...")

    if bot:
        await bot.close()

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

    # Broadcast weaving start
    await manager.broadcast({
        "type": "weaving_start",
        "data": {
            "query_id": query_id,
            "query_text": query_text,
            "timestamp": datetime.now().isoformat()
        }
    })

    # Simulate weaving steps (in real integration, these would come from orchestrator)
    steps = [
        "Loom Command",
        "Chrono Trigger",
        "Yarn Graph",
        "Resonance Shed",
        "Warp Space",
        "Convergence Engine",
        "Tool Execution",
        "Spacetime Fabric",
        "Reflection Buffer"
    ]

    for i, step_name in enumerate(steps, 1):
        # Broadcast step start
        await broadcast_weaving_step(
            query_id=query_id,
            query_text=query_text,
            step=i,
            name=step_name,
            status="in_progress"
        )

        # Simulate processing time
        await asyncio.sleep(0.05)

    # Execute actual weaving
    try:
        response: WeavingResponse = await bot.weave(
            query=query_text,
            user_id=user_id,
            room_id=room_id,
            complexity=complexity
        )

        # Broadcast completion
        for i, step_name in enumerate(steps, 1):
            await broadcast_weaving_step(
                query_id=query_id,
                query_text=query_text,
                step=i,
                name=step_name,
                status="completed",
                latency_ms=response.latency_ms / len(steps)  # Approximate
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

        # Add to query history
        query_record = {
            "query_id": query_id,
            "query_text": query_text,
            "timestamp": datetime.now().isoformat(),
            "confidence": response.confidence,
            "tool_used": response.tool_used,
            "latency_ms": response.latency_ms,
            "cache_hit": response.cache_hit,
            "response": response.text
        }
        query_history.append(query_record)

        # Broadcast final result
        await manager.broadcast({
            "type": "weaving_complete",
            "data": query_record
        })

        return {
            "success": True,
            "data": {
                "query_id": query_id,
                "response": response.text,
                "confidence": response.confidence,
                "tool_used": response.tool_used,
                "latency_ms": response.latency_ms,
                "cache_hit": response.cache_hit,
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


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "dashboard_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

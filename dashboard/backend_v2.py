#!/usr/bin/env python3
"""
🚀 mythRL Dashboard Backend v2 - MythRLShuttle Architecture
============================================================
Uses the new protocol-based MythRLShuttle with 244D semantic calculus.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add mythRL to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dev.protocol_modules_mythrl import MythRLShuttle, ComplexityLevel

app = FastAPI(title="mythRL Narrative Intelligence API v2", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global MythRLShuttle instance
shuttle: Optional[MythRLShuttle] = None


class QueryRequest(BaseModel):
    text: str
    complexity: Optional[str] = None  # "lite", "fast", "full", "research"


class QueryResponse(BaseModel):
    text: str
    complexity: str
    confidence: float
    execution_time_ms: float
    result: Dict[str, Any]
    provenance: Dict[str, Any]


@app.on_event("startup")
async def startup():
    """Initialize the MythRLShuttle on startup."""
    global shuttle
    print("🚀 Initializing MythRLShuttle...")
    shuttle = MythRLShuttle()
    print("✅ MythRLShuttle ready!")


@app.get("/")
async def root():
    """API health check."""
    return {
        "status": "online",
        "service": "mythRL Narrative Intelligence",
        "version": "2.0.0",
        "architecture": "MythRLShuttle",
        "features": [
            "protocol-based-architecture",
            "3-5-7-9-progressive-complexity",
            "244d-semantic-calculus",
            "multipass-memory-crawling",
            "full-provenance-tracing"
        ]
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_shuttle(request: QueryRequest):
    """
    Query the MythRLShuttle with progressive complexity.
    """
    if shuttle is None:
        return {"error": "Shuttle not initialized"}
    
    start_time = time.perf_counter()
    
    # Execute query through shuttle
    context = {}
    if request.complexity:
        context["requested_complexity"] = request.complexity
    
    result = await shuttle.weave(request.text, context)
    
    execution_time = (time.perf_counter() - start_time) * 1000
    
    return QueryResponse(
        text=request.text,
        complexity=result.complexity_level.name.lower(),
        confidence=result.confidence,
        execution_time_ms=execution_time,
        result=result.data,
        provenance={
            "operation_id": result.provenance.operation_id,
            "complexity_steps": result.complexity_level.value,
            "shuttle_events": len(result.provenance.shuttle_events),
            "protocol_calls": len(result.provenance.protocol_calls),
        }
    )


@app.get("/api/stats")
async def get_stats():
    """Get shuttle statistics."""
    if shuttle is None:
        return {"error": "Shuttle not initialized"}
    
    return {
        "status": "active",
        "protocols_registered": {
            "pattern_selection": shuttle.pattern_selection is not None,
            "decision_engine": shuttle.decision_engine is not None,
            "memory_backend": shuttle.memory_backend is not None,
            "feature_extraction": shuttle.feature_extraction is not None,
            "warp_space": shuttle.warp_space is not None,
            "tool_execution": shuttle.tool_execution is not None,
        },
        "complexity_levels": [level.name for level in ComplexityLevel]
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting mythRL API Server v2")
    print("📡 http://localhost:8000")
    print("📊 Docs: http://localhost:8000/docs")
    print("🎯 Architecture: MythRLShuttle + 244D Semantic Calculus")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

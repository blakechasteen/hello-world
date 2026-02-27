#!/usr/bin/env python3
"""
Squad Simple FastAPI Server
Simplified version without recursive learning dependencies
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# HoloLoom imports (simplified)
from hololoom.config import Config
from hololoom.documentation.types import Query, MemoryShard
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.fabric.spacetime import Spacetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================

class CodeContext(BaseModel):
    """Code context from VS Code"""
    currentFile: Optional[str] = None
    fileName: Optional[str] = None
    languageId: Optional[str] = None
    selection: Optional[str] = None
    diagnostics: Optional[List[Dict]] = None
    workspace: Optional[str] = None


class QueryRequest(BaseModel):
    """Query request from VS Code"""
    text: str
    context: Optional[CodeContext] = None
    mode: str = "direct"  # For now, all modes use direct
    max_steps: int = 5


class QueryResponse(BaseModel):
    """Query response to VS Code"""
    response: str
    confidence: float
    reasoning_mode: str
    steps_taken: List[Dict[str, Any]]
    total_queries: int
    total_duration_ms: float
    verification: Optional[Dict[str, Any]] = None


# ============================================================================
# Global State
# ============================================================================

app = FastAPI(title="Squad Server (Simple)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[WeavingOrchestrator] = None
config: Optional[Config] = None


# ============================================================================
# Lifecycle
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize HoloLoom on server startup"""
    global orchestrator, config

    logger.info("Starting Squad server (simple mode)...")

    # Create config
    config = Config.fast()

    # Create memory shards
    shards = create_initial_shards()

    # Create orchestrator
    orchestrator = WeavingOrchestrator(cfg=config, shards=shards)

    logger.info("Squad server ready! 🚀")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on server shutdown"""
    logger.info("Squad server stopped")


def create_initial_shards() -> List[MemoryShard]:
    """Create initial memory shards with coding knowledge"""
    return [
        MemoryShard(
            id="squad_001",
            text="Python programming language for general purpose programming",
            episode="squad:bootstrap",
            entities=["python", "programming"],
            motifs=["language", "code"],
            metadata={"source": "squad:bootstrap", "timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="squad_002",
            text="TypeScript programming language with static typing",
            episode="squad:bootstrap",
            entities=["typescript", "javascript"],
            motifs=["language", "code", "types"],
            metadata={"source": "squad:bootstrap", "timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="squad_003",
            text="VS Code extension API for building editor extensions",
            episode="squad:bootstrap",
            entities=["vscode", "extension", "api"],
            motifs=["development", "ide"],
            metadata={"source": "squad:bootstrap", "timestamp": datetime.now().isoformat()}
        )
    ]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "orchestrator_ready": orchestrator is not None,
        "mode": "simple",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Execute query using WeavingOrchestrator"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Create query
        query = Query(
            text=request.text,
            metadata={
                "source": "squad:vscode",
                "context": request.context.dict() if request.context else {}
            }
        )

        # Execute weaving
        logger.info(f"Query: {request.text[:100]}...")
        start_time = datetime.now()

        spacetime = await orchestrator.weave(query)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Format response
        response = QueryResponse(
            response=spacetime.metadata.get("response", "Query processed successfully"),
            confidence=spacetime.confidence,
            reasoning_mode="direct",
            steps_taken=[{
                "type": "weaving",
                "confidence": spacetime.confidence
            }],
            total_queries=1,
            total_duration_ms=duration_ms,
            verification=None
        )

        logger.info(f"Response confidence: {spacetime.confidence:.2f}")
        return response

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def handle_chat(request: Dict):
    """Simple chat endpoint"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        query = Query(
            text=request.get("message", ""),
            metadata={"source": "squad:chat"}
        )

        spacetime = await orchestrator.weave(query)

        return {
            "response": spacetime.metadata.get("response", "Query processed"),
            "confidence": spacetime.confidence
        }

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return {
        "status": "simple_mode",
        "message": "Running in simple mode (no recursive learning)"
    }


@app.post("/ingest/workspace")
async def ingest_workspace_endpoint(request: Dict):
    """
    Ingest VS Code workspace into HoloLoom memory

    Scans workspace directory and creates memory shards for all code files.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    workspace_path = request.get("workspace_path")
    if not workspace_path:
        raise HTTPException(status_code=400, detail="workspace_path required")

    try:
        from workspace_ingester import WorkspaceIngester

        logger.info(f"Ingesting workspace: {workspace_path}")

        ingester = WorkspaceIngester(workspace_path)
        shards = await ingester.ingest()

        stats = ingester.get_stats()
        stats["shards_created"] = len(shards)

        logger.info(f"Workspace ingested: {stats}")

        return {
            "status": "success",
            "stats": stats,
            "message": f"Ingested {stats['files_ingested']} files, created {len(shards)} shards"
        }

    except Exception as e:
        logger.error(f"Workspace ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server_simple:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )

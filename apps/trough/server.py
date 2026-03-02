#!/usr/bin/env python3
"""
Squad FastAPI Server
Exposes HoloLoom's agentic reasoning capabilities as a REST API for VS Code extension
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

# HoloLoom imports
from hololoom.config import Config
from hololoom.documentation.types import Query, MemoryShard
from hololoom.agentic import (
    create_agentic_orchestrator,
    AgenticOrchestrator,
    ReasoningMode,
    AgenticResult
)
from hololoom.alignment.audit_trail import AuditTrail

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
    mode: str = "verify"  # direct, verify, research, plan_execute
    max_steps: int = 5


class ChatRequest(BaseModel):
    """Chat request from VS Code"""
    message: str
    context: Optional[CodeContext] = None


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

app = FastAPI(title="Squad Server", version="0.1.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[AgenticOrchestrator] = None
config: Optional[Config] = None
audit_trail: Optional[AuditTrail] = None


# ============================================================================
# Lifecycle
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize HoloLoom on server startup"""
    global orchestrator, config, audit_trail

    logger.info("Starting Squad server...")

    # Create config
    config = Config.fast()  # Use FAST mode for good balance

    # Create audit trail
    audit_trail = AuditTrail()

    # Create memory shards (empty for now - will ingest code on demand)
    shards = create_initial_shards()

    # Create agentic orchestrator
    orchestrator = await create_agentic_orchestrator(
        config=config,
        shards=shards,
        enable_verification=True,
        enable_goal_tracking=True,
        audit_trail=audit_trail
    )

    logger.info("Squad server ready! 🚀")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on server shutdown"""
    global orchestrator

    if orchestrator:
        await orchestrator.close()

    logger.info("Squad server stopped")


def create_initial_shards() -> List[MemoryShard]:
    """Create initial memory shards with coding knowledge"""
    return [
        MemoryShard(
            content="Python programming language",
            entities=["python", "programming"],
            motifs=["language", "code"],
            timestamp=datetime.now(),
            source="squad:bootstrap"
        ),
        MemoryShard(
            content="TypeScript programming language",
            entities=["typescript", "javascript"],
            motifs=["language", "code", "types"],
            timestamp=datetime.now(),
            source="squad:bootstrap"
        ),
        MemoryShard(
            content="VS Code extension development",
            entities=["vscode", "extension", "api"],
            motifs=["development", "ide"],
            timestamp=datetime.now(),
            source="squad:bootstrap"
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
        "timestamp": datetime.now().isoformat()
    }


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Execute agentic query with reasoning

    This is the main endpoint for Squad queries from VS Code.
    Uses HoloLoom's AgenticOrchestrator for multi-step reasoning.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Parse reasoning mode
        mode_map = {
            "direct": ReasoningMode.DIRECT,
            "verify": ReasoningMode.VERIFY,
            "research": ReasoningMode.RESEARCH,
            "plan_execute": ReasoningMode.PLAN_EXECUTE
        }
        mode = mode_map.get(request.mode, ReasoningMode.VERIFY)

        # Create query with code context
        query = Query(
            text=request.text,
            metadata={
                "source": "squad:vscode",
                "context": request.context.dict() if request.context else {}
            }
        )

        # Execute agentic reasoning
        logger.info(f"Query: {request.text[:100]}... (mode={request.mode})")
        result = await orchestrator.reason(
            query=query,
            mode=mode,
            max_steps=request.max_steps
        )

        # Format response
        response = format_agentic_result(result)

        logger.info(f"Response confidence: {result.spacetime.confidence:.2f}")
        return response

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def handle_chat(request: ChatRequest):
    """
    Conversational chat endpoint

    Simpler endpoint for quick questions without full agentic reasoning.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Use DIRECT mode for chat (faster)
        query = Query(
            text=request.message,
            metadata={"source": "squad:chat"}
        )

        result = await orchestrator.reason(
            query=query,
            mode=ReasoningMode.DIRECT,
            max_steps=1
        )

        return {
            "response": result.spacetime.metadata.get("response", "No response generated"),
            "confidence": result.spacetime.confidence
        }

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    if not orchestrator or not audit_trail:
        return {"status": "not_ready"}

    # Get audit trail stats
    decisions = audit_trail.get_recent_decisions(limit=100)

    return {
        "total_queries": len(decisions),
        "avg_confidence": sum(d.metadata.get("confidence", 0) for d in decisions) / len(decisions) if decisions else 0,
        "reasoning_modes": {
            "direct": sum(1 for d in decisions if d.metadata.get("mode") == "direct"),
            "verify": sum(1 for d in decisions if d.metadata.get("mode") == "verify"),
            "research": sum(1 for d in decisions if d.metadata.get("mode") == "research"),
            "plan_execute": sum(1 for d in decisions if d.metadata.get("mode") == "plan_execute")
        }
    }


@app.post("/ingest/code")
async def ingest_code(files: List[Dict[str, str]]):
    """
    Ingest code files into HoloLoom memory

    Allows VS Code to feed code context into Squad's knowledge base.
    """
    # TODO: Implement code ingestion with CodeSpinner
    return {"status": "not_implemented", "message": "Code ingestion coming soon"}


# ============================================================================
# Helper Functions
# ============================================================================

def format_agentic_result(result: AgenticResult) -> QueryResponse:
    """Format AgenticResult for API response"""
    verification_dict = None
    if result.verification:
        verification_dict = {
            "verified": result.verification.verified,
            "confidence": result.verification.confidence,
            "contradictions": result.verification.contradictions,
            "supporting_evidence": result.verification.supporting_evidence,
            "suggested_refinements": result.verification.suggested_refinements
        }

    return QueryResponse(
        response=result.spacetime.metadata.get("response", "No response generated"),
        confidence=result.spacetime.confidence,
        reasoning_mode=result.reasoning_mode.value,
        steps_taken=result.steps_taken,
        total_queries=result.total_queries,
        total_duration_ms=result.total_duration_ms,
        verification=verification_dict
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

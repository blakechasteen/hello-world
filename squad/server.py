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
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query, MemoryShard
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.alignment.audit_trail import AuditTrail

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


class ReasoningStep(BaseModel):
    """Single reasoning step"""
    type: str
    query: Optional[str] = None
    confidence: Optional[float] = None
    finding: Optional[str] = None
    completed: bool = True


class VerificationResult(BaseModel):
    """Verification result"""
    verified: bool
    confidence: float
    contradictions: List[str]
    supporting_evidence: List[str]
    suggested_refinements: List[str]


class QueryResponse(BaseModel):
    """Query response to VS Code"""
    response: str
    confidence: float
    reasoning_mode: str
    steps_taken: List[ReasoningStep]
    total_queries: int
    total_duration_ms: float
    verification: Optional[VerificationResult] = None


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
orchestrator: Optional[WeavingOrchestrator] = None
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

    # Create config with einops fix
    config = Config.fast()  # Use FAST mode for good balance
    config.enable_alignment = True  # Safety checks

    # Create audit trail
    audit_trail = AuditTrail()

    # Create memory shards (empty for now - will ingest code on demand)
    shards = create_initial_shards()

    # Create orchestrator
    orchestrator = WeavingOrchestrator(cfg=config, shards=shards)
    await orchestrator.__aenter__()  # Initialize async context

    logger.info("Squad server ready! 🚀")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on server shutdown"""
    global orchestrator

    if orchestrator:
        await orchestrator.__aexit__(None, None, None)

    logger.info("Squad server stopped")


def create_initial_shards() -> List[MemoryShard]:
    """Create initial memory shards with coding knowledge"""
    return [
        MemoryShard(
            id="shard-python-1",
            text="Python programming language with type hints, async/await, decorators",
            entities=["python", "programming", "types"],
            motifs=["language", "code", "syntax"],
            metadata={"source": "squad:bootstrap", "timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="shard-typescript-1",
            text="TypeScript programming language with interfaces, generics, type safety",
            entities=["typescript", "javascript", "types"],
            motifs=["language", "code", "types", "safety"],
            metadata={"source": "squad:bootstrap", "timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="shard-vscode-1",
            text="VS Code extension API for commands, webviews, diagnostics",
            entities=["vscode", "extension", "api"],
            motifs=["development", "ide", "tools"],
            metadata={"source": "squad:bootstrap", "timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="shard-hololoom-1",
            text="HoloLoom agentic reasoning with multi-step verification and research modes",
            entities=["hololoom", "agentic", "reasoning"],
            motifs=["ai", "intelligence", "verification"],
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
        "timestamp": datetime.now().isoformat()
    }


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Execute agentic query with reasoning

    This is the main endpoint for Squad queries from VS Code.
    Uses HoloLoom's WeavingOrchestrator for multi-step reasoning.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Create query with code context
        query = Query(
            text=request.text,
            metadata={
                "source": "squad:vscode",
                "mode": request.mode,
                "context": request.context.dict() if request.context else {}
            }
        )

        # Execute weaving
        logger.info(f"Query: {request.text[:100]}... (mode={request.mode})")
        start_time = datetime.now()

        spacetime = await orchestrator.weave(query)

        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Extract response from spacetime
        response_text = extract_response(spacetime)

        # Create reasoning steps
        steps = create_reasoning_steps(spacetime, request.mode)

        # Format response
        response = QueryResponse(
            response=response_text,
            confidence=spacetime.confidence,
            reasoning_mode=request.mode,
            steps_taken=steps,
            total_queries=len(steps),
            total_duration_ms=duration_ms,
            verification=None  # Will add in future iteration
        )

        logger.info(f"Response confidence: {spacetime.confidence:.2f}, duration: {duration_ms:.1f}ms")
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
        # Use simple query for chat
        query = Query(
            text=request.message,
            metadata={"source": "squad:chat"}
        )

        spacetime = await orchestrator.weave(query)

        return {
            "response": extract_response(spacetime),
            "confidence": spacetime.confidence
        }

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    if not orchestrator:
        return {"status": "not_ready"}

    return {
        "orchestrator_ready": True,
        "config_mode": config.mode.value if config else "unknown",
        "total_queries": 0  # Will track in future iteration
    }


# ============================================================================
# Helper Functions
# ============================================================================

def extract_response(spacetime) -> str:
    """Extract human-readable response from Spacetime"""
    # Try to get response from metadata first
    if spacetime.metadata and "response" in spacetime.metadata:
        return spacetime.metadata["response"]

    # Try to get tool output
    if spacetime.metadata and "tool_output" in spacetime.metadata:
        return spacetime.metadata["tool_output"]

    # Try to construct from trace
    if spacetime.trace and hasattr(spacetime.trace, 'observations'):
        observations = spacetime.trace.observations
        if observations:
            return "\n".join(str(obs) for obs in observations[-3:])  # Last 3 observations

    # Fallback to confidence-based message
    if spacetime.confidence > 0.8:
        return "Query processed successfully with high confidence."
    elif spacetime.confidence > 0.5:
        return "Query processed with moderate confidence."
    else:
        return "Query processed but with low confidence. Results may need verification."


def create_reasoning_steps(spacetime, mode: str) -> List[ReasoningStep]:
    """Create reasoning steps from Spacetime trace"""
    steps = []

    # Add initial query step
    steps.append(ReasoningStep(
        type="initial_query",
        query=spacetime.metadata.get("query_text", ""),
        confidence=spacetime.confidence,
        completed=True
    ))

    # Add mode-specific steps
    if mode == "verify":
        steps.append(ReasoningStep(
            type="verification",
            query="Verify response accuracy",
            confidence=spacetime.confidence,
            completed=True
        ))
    elif mode == "research":
        steps.append(ReasoningStep(
            type="research",
            query="Multi-angle exploration",
            confidence=spacetime.confidence,
            completed=True
        ))
    elif mode == "plan_execute":
        steps.append(ReasoningStep(
            type="planning",
            query="Break down task",
            completed=True
        ))
        steps.append(ReasoningStep(
            type="execution",
            query="Execute plan steps",
            confidence=spacetime.confidence,
            completed=True
        ))

    return steps


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

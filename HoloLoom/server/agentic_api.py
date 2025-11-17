#!/usr/bin/env python3
"""
HoloLoom Agentic API Server
============================
FastAPI server for agentic intelligence integration with VS Code Squad extension.

Provides HTTP endpoints that match the TypeScript interfaces in squad/src/HoloLoomBridge.ts

Usage:
    # Development
    uvicorn HoloLoom.server.agentic_api:app --reload --port 8000

    # Production
    uvicorn HoloLoom.server.agentic_api:app --host 0.0.0.0 --port 8000 --workers 4
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from HoloLoom.agentic import AgenticResult, ReasoningMode, create_agentic_orchestrator
from HoloLoom.alignment.audit_trail import AuditTrail
from HoloLoom.config import Config
from HoloLoom.documentation.types import MemoryShard, Query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models (Match TypeScript interfaces)
# ============================================================================


class CodeContext(BaseModel):
    """Code context from VS Code editor (matches TypeScript interface)."""

    currentFile: str | None = None
    fileName: str | None = None
    languageId: str | None = None
    selection: str | None = None
    workspace: str | None = None
    diagnostics: list[dict] | None = None


class QueryRequest(BaseModel):
    """Query request from VS Code extension."""

    text: str = Field(..., description="Query text")
    context: CodeContext | None = Field(None, description="Code context from editor")
    mode: str = Field(
        "verify", description="Reasoning mode: direct, verify, research, plan_execute"
    )
    max_steps: int = Field(5, description="Maximum reasoning steps")


class VerificationResponse(BaseModel):
    """Verification result (matches TypeScript interface)."""

    verified: bool
    confidence: float
    contradictions: list[str]
    supporting_evidence: list[str]
    suggested_refinements: list[str]


class ReasoningStepResponse(BaseModel):
    """Single reasoning step (matches TypeScript interface)."""

    type: str
    query: str | None = None
    confidence: float | None = None
    finding: str | None = None
    completed: bool | None = None
    tool: str | None = None


class AgenticResponse(BaseModel):
    """
    Agentic reasoning result (matches TypeScript AgenticResult interface).

    This is the main response returned to the VS Code extension.
    """

    response: str
    confidence: float
    reasoning_mode: str
    steps_taken: list[ReasoningStepResponse]
    total_queries: int
    total_duration_ms: float
    verification: VerificationResponse | None = None

    # Additional metadata
    timestamp: str
    query_id: str


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="HoloLoom Agentic API",
    description="Agentic intelligence backend for VS Code Squad extension",
    version="1.0.0",
)

# CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global State
# ============================================================================


class ServerState:
    """Global server state."""

    orchestrator: Any | None = None
    audit_trail: AuditTrail | None = None
    config: Config | None = None
    shards: list[MemoryShard] = []
    memory_backend: Any | None = None  # Persistent memory backend
    start_time: datetime | None = None  # Server start time for uptime tracking
    query_count: int = 0  # Total queries processed


state = ServerState()


# ============================================================================
# Lifecycle
# ============================================================================


@app.on_event("startup")
async def startup():
    """Initialize server with persistent memory."""
    logger.info("Starting HoloLoom Agentic API server...")

    # Track server start time for uptime metrics
    state.start_time = datetime.now()

    # Load config
    state.config = Config.fast()
    state.config.enable_agentic_reasoning = True

    # Initialize audit trail
    state.audit_trail = AuditTrail(persist_path="./alignment_logs")

    # ✅ Create persistent memory backend
    try:
        from HoloLoom.config import MemoryBackend
        from HoloLoom.memory.backend_factory import create_memory_backend

        state.config.memory_backend = MemoryBackend.HYBRID  # Use persistent storage
        state.memory_backend = await create_memory_backend(state.config)
        logger.info(f"Memory backend: {state.config.memory_backend.value}")

        # Load existing memories from persistent storage
        state.shards = await _load_from_persistent_backend()
        logger.info(f"Loaded {len(state.shards)} memories from persistent storage")

    except Exception as e:
        logger.warning(f"Persistent backend unavailable: {e}")
        logger.info("Falling back to in-memory storage")
        state.shards = _load_memory_shards()  # Use example shards as fallback

    # Create orchestrator (lazy init - see get_orchestrator)
    logger.info("HoloLoom server ready!")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down HoloLoom server...")
    if state.orchestrator:
        await state.orchestrator.close()


# ============================================================================
# Helper Functions
# ============================================================================


def _load_memory_shards() -> list[MemoryShard]:
    """
    Load memory shards from data source (fallback when persistent backend unavailable).

    Returns:
        Example shards for development/fallback
    """
    # Example: Load from a knowledge base
    return [
        MemoryShard(
            id="example_1",
            text="HoloLoom is a neural decision-making system with multi-scale embeddings.",
            episode="hololoom_basics",
            entities=["HoloLoom", "embeddings"],
            motifs=["definition"],
        )
    ]


async def _load_from_persistent_backend() -> list[MemoryShard]:
    """
    Load memories from persistent backend (Neo4j/Qdrant).

    Returns:
        List of MemoryShard objects loaded from storage
    """
    if not state.memory_backend:
        return []

    try:
        # For HYBRID backend, retrieve all stored memories
        # Note: HYBRID backend (Neo4j + Qdrant) uses unified memory protocol
        from HoloLoom.memory.protocol import MemoryQuery

        query = MemoryQuery(
            text="",
            limit=1000,  # Empty query = retrieve all  # Adjust based on your needs
        )

        result = await state.memory_backend.retrieve(query)

        # Convert Memory objects to MemoryShard objects
        shards = []
        for memory in result.memories:
            shard = MemoryShard(
                id=memory.id,
                text=memory.text,
                episode=memory.context.get("episode", "default"),
                entities=memory.context.get("entities", []),
                motifs=memory.context.get("motifs", []),
                metadata=memory.metadata,
            )
            shards.append(shard)

        return shards

    except Exception as e:
        logger.error(f"Failed to load from persistent backend: {e}")
        return []


async def get_orchestrator():
    """Get or create global orchestrator instance."""
    if state.orchestrator is None:
        logger.info("Initializing agentic orchestrator...")
        state.orchestrator = await create_agentic_orchestrator(
            state.config,
            state.shards,
            enable_verification=True,
            enable_goal_tracking=True,
            audit_trail=state.audit_trail,
        )
        logger.info("Orchestrator ready!")

    return state.orchestrator


def _format_verification(verification) -> VerificationResponse | None:
    """Format verification result for API response."""
    if verification is None:
        return None

    return VerificationResponse(
        verified=verification.verified,
        confidence=verification.confidence,
        contradictions=verification.contradictions,
        supporting_evidence=verification.supporting_evidence,
        suggested_refinements=verification.suggested_refinements,
    )


def _format_steps(steps: list[dict]) -> list[ReasoningStepResponse]:
    """Format reasoning steps for API response."""
    return [
        ReasoningStepResponse(
            type=step.get("type", "unknown"),
            query=step.get("query"),
            confidence=step.get("confidence"),
            finding=step.get("finding"),
            completed=step.get("completed"),
            tool=step.get("tool_used") or step.get("tool"),
        )
        for step in steps
    ]


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Used by VS Code extension to verify server is running.
    """
    return {
        "status": "ok",
        "service": "HoloLoom Agentic API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/query", response_model=AgenticResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint for agentic reasoning.

    Matches VS Code extension's HoloLoomBridge.query() expectations.

    Args:
        request: QueryRequest with text, context, mode, max_steps

    Returns:
        AgenticResponse with reasoning results

    Example:
        POST /query
        {
          "text": "Explain this TypeScript code",
          "context": {
            "languageId": "typescript",
            "fileName": "example.ts",
            "selection": "function foo() { ... }"
          },
          "mode": "verify",
          "max_steps": 5
        }
    """
    start_time = datetime.now()

    try:
        # Get orchestrator
        orchestrator = await get_orchestrator()

        # Map mode string to enum
        mode_map = {
            "direct": ReasoningMode.DIRECT,
            "verify": ReasoningMode.VERIFY,
            "research": ReasoningMode.RESEARCH,
            "plan_execute": ReasoningMode.PLAN_EXECUTE,
        }
        mode = mode_map.get(request.mode, ReasoningMode.VERIFY)

        # Create query
        query = Query(text=request.text)

        # Add code context to metadata
        if request.context:
            query.metadata = {
                "code_context": request.context.dict(),
                "language": request.context.languageId,
                "file": request.context.fileName,
                "selection": request.context.selection,
                "workspace": request.context.workspace,
            }

        # Run agentic reasoning
        logger.info(f"Query: {request.text[:100]}... (mode={request.mode})")
        result: AgenticResult = await orchestrator.reason(
            query, mode=mode, max_steps=request.max_steps
        )

        # Track successful query
        state.query_count += 1

        # Extract response text
        response_text = result.spacetime.metadata.get("response", "")
        if not response_text:
            # Fallback: use context or generate generic response
            response_text = f"Processed query with {result.reasoning_mode.value} mode."

        # Format response (matches TypeScript AgenticResult interface)
        return AgenticResponse(
            response=response_text,
            confidence=result.spacetime.confidence,
            reasoning_mode=result.reasoning_mode.value,
            steps_taken=_format_steps(result.steps_taken),
            total_queries=result.total_queries,
            total_duration_ms=result.total_duration_ms,
            verification=_format_verification(result.verification),
            timestamp=start_time.isoformat(),
            query_id=result.spacetime.query_id,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get server statistics.

    Returns:
        Statistics about queries processed, audit trail, etc.
    """
    # Calculate uptime
    uptime = "unknown"
    if state.start_time:
        delta = datetime.now() - state.start_time
        hours = delta.total_seconds() / 3600
        if hours < 1:
            minutes = delta.total_seconds() / 60
            uptime = f"{minutes:.1f} minutes"
        elif hours < 24:
            uptime = f"{hours:.1f} hours"
        else:
            days = hours / 24
            uptime = f"{days:.1f} days"

    stats = {
        "server_uptime": uptime,
        "total_queries": state.query_count,
        "orchestrator_ready": state.orchestrator is not None,
        "memory_shards": len(state.shards),
    }

    if state.audit_trail:
        stats["audit_trail_entries"] = len(state.audit_trail.logs)

    return stats


@app.get("/audit-trail")
async def get_audit_trail(limit: int = 100):
    """
    Get recent audit trail entries.

    Args:
        limit: Maximum number of entries to return

    Returns:
        Recent audit trail entries
    """
    if not state.audit_trail:
        return {"entries": []}

    # Get recent entries
    entries = state.audit_trail.logs[-limit:]

    return {
        "total": len(state.audit_trail.logs),
        "entries": [
            {
                "decision_id": log.decision_id,
                "decision_type": log.decision_type.value,
                "outcome": log.outcome.value,
                "reason": log.reason,
                "timestamp": log.timestamp.isoformat(),
                "confidence": log.confidence,
            }
            for log in entries
        ],
    }


@app.post("/memories/add")
async def add_memory(memory: dict):
    """
    Add new memory to persistent storage.

    Args:
        memory: Dict with text, episode, entities, motifs, metadata

    Returns:
        Success status and memory ID

    Example:
        POST /memories/add
        {
          "text": "Thompson Sampling balances exploration and exploitation",
          "episode": "algorithms",
          "entities": ["Thompson Sampling"],
          "motifs": ["definition"],
          "metadata": {"topic": "ML", "confidence": 0.9}
        }
    """
    try:
        if not state.memory_backend:
            return {
                "success": False,
                "message": "Persistent backend not available",
                "memory_id": None,
            }

        from HoloLoom.memory.protocol import Memory

        # Create Memory object
        new_memory = Memory(
            id=f"mem_{datetime.now().timestamp()}",
            text=memory.get("text", ""),
            context={
                "episode": memory.get("episode", "default"),
                "entities": memory.get("entities", []),
                "motifs": memory.get("motifs", []),
            },
            metadata=memory.get("metadata", {}),
        )

        # Store in persistent backend
        await state.memory_backend.store([new_memory])

        # Also add to in-memory shards for immediate availability
        shard = MemoryShard(
            id=new_memory.id,
            text=new_memory.text,
            episode=new_memory.context.get("episode", "default"),
            entities=new_memory.context.get("entities", []),
            motifs=new_memory.context.get("motifs", []),
            metadata=new_memory.metadata,
        )
        state.shards.append(shard)

        logger.info(f"Added memory: {new_memory.id}")

        return {"success": True, "message": "Memory added successfully", "memory_id": new_memory.id}

    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting HoloLoom Agentic API server (development mode)...")
    uvicorn.run(
        "HoloLoom.server.agentic_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )

#!/usr/bin/env python3
"""
HoloLoom Agentic API - Fully Integrated (LLM + Memory + UI)
============================================================
Complete implementation with:
- Real LLM calls (Ollama/Anthropic)
- Persistent memory (Neo4j + Qdrant)
- WebSocket support for UI streaming
- Background learning

Usage:
    # Local development (Ollama)
    python HoloLoom/server/agentic_api_integrated.py

    # Production (Anthropic)
    ANTHROPIC_API_KEY=sk-ant-xxx python HoloLoom/server/agentic_api_integrated.py

Then open:
    - REST API: http://localhost:8000/docs
    - Chat UI: http://localhost:8000 (if serving static files)
"""

import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# HoloLoom imports
from HoloLoom.agentic import create_agentic_orchestrator, ReasoningMode
from HoloLoom.config import Config, MemoryBackend, Environment
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.alignment.audit_trail import AuditTrail
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.memory.protocol import Memory, MemoryQuery
from HoloLoom.awareness.llm_integration import OllamaLLM, AnthropicLLM

# Use LLM-integrated orchestrator
from HoloLoom.weaving_orchestrator_llm import WeavingOrchestrator as LLMOrchestrator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Environment Detection
# ============================================================================

def _detect_environment() -> Environment:
    """
    Detect deployment environment from HOLOLOOM_ENV environment variable.
    
    Returns:
        Environment enum (DEVELOPMENT, STAGING, or PRODUCTION)
        Defaults to DEVELOPMENT if not set.
    """
    env_str = os.getenv('HOLOLOOM_ENV', 'development').lower()
    
    if env_str in ('development', 'dev'):
        return Environment.DEVELOPMENT
    elif env_str in ('staging', 'stage'):
        return Environment.STAGING
    elif env_str in ('production', 'prod'):
        return Environment.PRODUCTION
    else:
        logger.warning(f"Unknown HOLOLOOM_ENV value: {env_str}, defaulting to DEVELOPMENT")
        return Environment.DEVELOPMENT



# ============================================================================
# Environment Detection
# ============================================================================

def _detect_environment() -> Environment:
    """
    Detect deployment environment from HOLOLOOM_ENV environment variable.
    
    Returns:
        Environment enum (DEVELOPMENT, STAGING, or PRODUCTION)
        Defaults to DEVELOPMENT if not set.
    """
    env_str = os.getenv('HOLOLOOM_ENV', 'development').lower()
    
    if env_str in ('development', 'dev'):
        return Environment.DEVELOPMENT
    elif env_str in ('staging', 'stage'):
        return Environment.STAGING
    elif env_str in ('production', 'prod'):
        return Environment.PRODUCTION
    else:
        logger.warning(f"Unknown HOLOLOOM_ENV value: {env_str}, defaulting to DEVELOPMENT")
        return Environment.DEVELOPMENT




# ============================================================================
# Request/Response Models
# ============================================================================

class CodeContext(BaseModel):
    """Code context from VS Code editor."""
    currentFile: Optional[str] = None
    fileName: Optional[str] = None
    languageId: Optional[str] = None
    selection: Optional[str] = None
    workspace: Optional[str] = None


class QueryRequest(BaseModel):
    """Query request."""
    text: str = Field(..., description="Query text")
    context: Optional[CodeContext] = None
    mode: str = Field("verify", description="Reasoning mode")
    max_steps: int = Field(5, description="Maximum steps")


class AgenticResponse(BaseModel):
    """Agentic result response."""
    response: str
    confidence: float
    reasoning_mode: str
    steps_taken: List[Dict]
    total_queries: int
    total_duration_ms: float
    verification: Optional[Dict] = None
    timestamp: str
    query_id: str
    llm_provider: Optional[str] = None  # ✅ Which LLM was used
    llm_model: Optional[str] = None


class MemoryAddRequest(BaseModel):
    """Add memory request."""
    text: str
    entities: List[str] = []
    motifs: List[str] = []
    episode: str = "default"
    metadata: Dict = {}


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="HoloLoom Agentic API (Integrated)",
    description="Full LLM + Memory + Agentic Intelligence",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global State
# ============================================================================

class ServerState:
    """Global server state."""
    orchestrator: Optional[Any] = None
    llm: Optional[Any] = None
    memory_backend: Optional[Any] = None
    audit_trail: Optional[AuditTrail] = None
    config: Optional[Config] = None
    shards: List[MemoryShard] = []
    active_websockets: List[WebSocket] = []


state = ServerState()


# ============================================================================
# Lifecycle
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize server with full integration."""
    logger.info("="*80)
    logger.info("Starting HoloLoom Agentic API (Fully Integrated)")
    logger.info("="*80)

    # 1. Initialize LLM
    logger.info("\n[1/5] Initializing LLM...")
    state.llm = await _init_llm()

    # 2. Initialize persistent memory
    logger.info("\n[2/5] Initializing persistent memory...")
    state.memory_backend = await _init_memory()

    # 3. Load existing memories
    logger.info("\n[3/5] Loading memories from persistent storage...")
    state.shards = await _load_memories()

    # 4. Initialize config
    logger.info("\n[4/5] Initializing config...")
    state.config = Config.fast()

    # Detect and set environment
    detected_env = _detect_environment()
    state.config.environment = detected_env
    logger.info(f"  Environment: {detected_env.value} (from HOLOLOOM_ENV)")
    logger.info(f"  Safety mode: {'auto-approve all' if state.config.safety_testing_mode else 'require approvals'}")

    state.config.enable_agentic_reasoning = True
    state.config.memory_backend = MemoryBackend.HYBRID

    # 5. Initialize audit trail
    logger.info("\n[5/5] Initializing audit trail...")
    state.audit_trail = AuditTrail(persist_path="./alignment_logs")

    logger.info("\n" + "="*80)
    logger.info("✓ HoloLoom server ready!")
    logger.info(f"  - LLM: {state.llm.model if state.llm else 'None (fallback)'}")
    logger.info(f"  - Memory: {state.config.memory_backend.value if state.memory_backend else 'In-memory'}")
    logger.info(f"  - Shards loaded: {len(state.shards)}")
    logger.info("="*80)


async def _init_llm():
    """Initialize LLM (Ollama or Anthropic)."""
    # Try Anthropic first (if API key set)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            llm = AnthropicLLM(api_key=anthropic_key, model="claude-3-5-sonnet-20241022")
            logger.info("  ✓ Anthropic LLM initialized (Claude 3.5 Sonnet)")
            return llm
        except Exception as e:
            logger.warning(f"  ⚠ Anthropic init failed: {e}")

    # Fall back to Ollama
    try:
        llm = OllamaLLM(model="llama3.2:3b")
        if llm.is_available():
            logger.info("  ✓ Ollama LLM initialized (llama3.2:3b)")
            return llm
        else:
            logger.warning("  ⚠ Ollama not running (install from https://ollama.ai)")
            return None
    except Exception as e:
        logger.warning(f"  ⚠ Ollama init failed: {e}")
        return None


async def _init_memory():
    """Initialize persistent memory backend."""
    try:
        config = Config.fast()
        config.environment = _detect_environment()  # Set environment for safety
        config.memory_backend = MemoryBackend.HYBRID  # Neo4j + Qdrant with auto-fallback

        backend = await create_memory_backend(config)
        logger.info(f"  ✓ Memory backend initialized ({config.memory_backend.value})")
        return backend

    except Exception as e:
        logger.warning(f"  ⚠ Persistent memory unavailable: {e}")
        logger.info("  → Falling back to in-memory storage")
        return None


async def _load_memories() -> List[MemoryShard]:
    """Load memories from persistent backend."""
    if not state.memory_backend:
        logger.info("  → No persistent backend, starting with empty memory")
        return []

    try:
        # Query all memories (adjust limit as needed)
        query = MemoryQuery(text="", limit=1000)
        result = await state.memory_backend.retrieve(query)

        # Convert to shards
        shards = [
            MemoryShard(
                id=mem.id,
                text=mem.text,
                episode=mem.context.get("episode", "default"),
                entities=mem.context.get("entities", []),
                motifs=mem.context.get("motifs", []),
                metadata=mem.metadata
            )
            for mem in result.memories
        ]

        logger.info(f"  ✓ Loaded {len(shards)} memories from persistent storage")
        return shards

    except Exception as e:
        logger.error(f"  ✗ Failed to load memories: {e}")
        return []


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")
    if state.orchestrator:
        await state.orchestrator.close()

    # Close all WebSockets
    for ws in state.active_websockets:
        try:
            await ws.close()
        except:
            pass


# ============================================================================
# Helper Functions
# ============================================================================

async def get_orchestrator():
    """Get or create orchestrator with LLM integration."""
    if state.orchestrator is None:
        logger.info("Creating agentic orchestrator with LLM...")

        state.orchestrator = await create_agentic_orchestrator(
            state.config,
            state.shards,
            enable_verification=True,
            enable_goal_tracking=True,
            audit_trail=state.audit_trail
        )

        # Inject LLM into orchestrator
        if state.llm:
            # The orchestrator uses FullLearningEngine which uses WeavingOrchestrator
            # We need to swap in the LLM-enabled version
            # This is a bit hacky but works for now
            state.orchestrator.learning_engine.llm = state.llm

        logger.info("✓ Orchestrator ready with LLM integration")

    return state.orchestrator


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with status."""
    return {
        "service": "HoloLoom Agentic API (Integrated)",
        "version": "2.0.0",
        "status": "ready",
        "llm": state.llm.model if state.llm else "None",
        "memory": "persistent" if state.memory_backend else "in-memory",
        "memories_loaded": len(state.shards)
    }


@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "ok",
        "service": "HoloLoom Agentic API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "llm": "available" if state.llm else "unavailable",
            "memory": "persistent" if state.memory_backend else "in-memory",
            "orchestrator": "initialized" if state.orchestrator else "lazy-init"
        }
    }


@app.post("/query", response_model=AgenticResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint with LLM + memory integration.

    Now returns REAL LLM responses using persistent memory!
    """
    start_time = datetime.now()

    try:
        orchestrator = await get_orchestrator()

        # Map mode
        mode_map = {
            "direct": ReasoningMode.DIRECT,
            "verify": ReasoningMode.VERIFY,
            "research": ReasoningMode.RESEARCH,
            "plan_execute": ReasoningMode.PLAN_EXECUTE,
        }
        mode = mode_map.get(request.mode, ReasoningMode.VERIFY)

        # Create query
        query = Query(text=request.text)
        if request.context:
            query.metadata = {
                "code_context": request.context.dict(),
                "language": request.context.languageId,
            }

        # Run agentic reasoning with LLM
        logger.info(f"Processing query: {request.text[:80]}... (mode={request.mode})")
        result = await orchestrator.reason(query, mode=mode, max_steps=request.max_steps)

        # Extract LLM info from result
        llm_provider = result.spacetime.metadata.get("llm_provider")
        llm_model = result.spacetime.metadata.get("llm_model")

        # Build response
        response = AgenticResponse(
            response=result.spacetime.metadata.get("response", "No response generated"),
            confidence=result.spacetime.confidence,
            reasoning_mode=result.reasoning_mode.value,
            steps_taken=[
                {
                    "type": step.get("type"),
                    "query": step.get("query"),
                    "confidence": step.get("confidence"),
                    "tool": step.get("tool_used") or step.get("tool")
                }
                for step in result.steps_taken
            ],
            total_queries=result.total_queries,
            total_duration_ms=result.total_duration_ms,
            verification={
                "verified": result.verification.verified,
                "contradictions": result.verification.contradictions,
                "supporting_evidence": result.verification.supporting_evidence
            } if result.verification else None,
            timestamp=start_time.isoformat(),
            query_id=result.spacetime.query_id,
            llm_provider=llm_provider,
            llm_model=llm_model
        )

        logger.info(f"✓ Query complete: {result.total_duration_ms:.1f}ms, confidence={result.spacetime.confidence:.3f}")
        return response

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories/add")
async def add_memory(memory: MemoryAddRequest):
    """Add new memory to persistent storage."""
    try:
        # Create memory object
        mem = Memory(
            id=f"mem_{int(datetime.now().timestamp())}",
            text=memory.text,
            timestamp=datetime.now(),
            context={
                "entities": memory.entities,
                "motifs": memory.motifs,
                "episode": memory.episode
            },
            metadata=memory.metadata
        )

        # Store in persistent backend (if available)
        if state.memory_backend:
            await state.memory_backend.store([mem])
            logger.info(f"✓ Stored memory: {mem.id}")

        # Add to active shards
        shard = MemoryShard(
            id=mem.id,
            text=mem.text,
            episode=mem.context.get("episode", "default"),
            entities=mem.context.get("entities", []),
            motifs=mem.context.get("motifs", [])
        )
        state.shards.append(shard)

        return {"status": "success", "id": mem.id, "persistent": state.memory_backend is not None}

    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Server statistics."""
    return {
        "orchestrator_ready": state.orchestrator is not None,
        "llm_available": state.llm is not None,
        "llm_model": state.llm.model if state.llm else None,
        "memory_backend": state.config.memory_backend.value if state.memory_backend else "in-memory",
        "memories_loaded": len(state.shards),
        "audit_trail_entries": len(state.audit_trail.logs) if state.audit_trail else 0,
        "active_websockets": len(state.active_websockets)
    }


# ============================================================================
# WebSocket for UI Streaming
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time streaming to UI."""
    await websocket.accept()
    state.active_websockets.append(websocket)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            # Process query
            query_text = data.get("query", "")
            mode = data.get("mode", "verify")

            # Send "thinking" status
            await websocket.send_json({
                "type": "status",
                "message": f"Processing with {mode} mode..."
            })

            # Get response
            orchestrator = await get_orchestrator()
            mode_enum = {
                "direct": ReasoningMode.DIRECT,
                "verify": ReasoningMode.VERIFY,
                "research": ReasoningMode.RESEARCH,
                "plan_execute": ReasoningMode.PLAN_EXECUTE,
            }.get(mode, ReasoningMode.VERIFY)

            result = await orchestrator.reason(
                Query(text=query_text),
                mode=mode_enum
            )

            # Send result
            await websocket.send_json({
                "type": "result",
                "response": result.spacetime.metadata.get("response", ""),
                "confidence": result.spacetime.confidence,
                "reasoning_mode": result.reasoning_mode.value,
                "steps": result.total_queries,
                "verification": {
                    "verified": result.verification.verified,
                    "contradictions": result.verification.contradictions
                } if result.verification else None
            })

    except WebSocketDisconnect:
        state.active_websockets.remove(websocket)
        logger.info("WebSocket disconnected")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting HoloLoom Agentic API (Fully Integrated)...")
    logger.info("Features:")
    logger.info("  ✓ Real LLM calls (Ollama/Anthropic)")
    logger.info("  ✓ Persistent memory (Neo4j + Qdrant)")
    logger.info("  ✓ Agentic reasoning (4 modes)")
    logger.info("  ✓ WebSocket streaming for UI")
    logger.info("")

    uvicorn.run(
        "HoloLoom.server.agentic_api_integrated:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

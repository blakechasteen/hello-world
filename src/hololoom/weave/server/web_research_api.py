#!/usr/bin/env python3
"""
HoloLoom Web Research API
=========================
FastAPI server with web-enhanced agentic research + streaming support.

Features:
- /research/web - Web-enhanced research with citations
- /research/stream - Streaming endpoint (SSE)
- /conversations - Conversational threading
- Complete Perplexity-style experience

Usage:
    # Development
    uvicorn HoloLoom.server.web_research_api:app --reload --port 8000

    # Production
    uvicorn HoloLoom.server.web_research_api:app --host 0.0.0.0 --port 8000 --workers 4
"""

import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from HoloLoom.agentic import WebResearchOrchestrator, ReasoningMode
from HoloLoom.config import Config
from HoloLoom.Documentation.types import MemoryShard
from HoloLoom.alignment.audit_trail import AuditTrail


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class WebResearchRequest(BaseModel):
    """Request for web-enhanced research."""
    query: str = Field(..., description="Research query")
    max_web_results: int = Field(10, description="Max web results to retrieve")
    max_steps: int = Field(5, description="Max reasoning steps")
    enable_citations: bool = Field(True, description="Add inline citations")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for threading")


class WebResearchResponse(BaseModel):
    """Response from web research."""
    response: str
    cited_response: str
    confidence: float
    web_results_count: int
    memory_shards_count: int
    total_duration_ms: float
    web_search_time_ms: float
    citations: List[Dict[str, Any]]
    conversation_id: Optional[str] = None


class ConversationMessage(BaseModel):
    """Single message in conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationRequest(BaseModel):
    """Request with conversation context."""
    query: str
    conversation_id: Optional[str] = None
    include_history: bool = True


# ============================================================================
# Conversational Threading
# ============================================================================

class ConversationManager:
    """
    Manages multi-turn conversations with context preservation.

    Features:
    - Thread-aware queries
    - Context window management
    - Conversation persistence
    """

    def __init__(self, max_history: int = 10):
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        self.max_history = max_history

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ):
        """Add message to conversation."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        self.conversations[conversation_id].append(message)

        # Trim to max history
        if len(self.conversations[conversation_id]) > self.max_history * 2:
            self.conversations[conversation_id] = self.conversations[conversation_id][-(self.max_history * 2):]

    def get_history(self, conversation_id: str) -> List[ConversationMessage]:
        """Get conversation history."""
        return self.conversations.get(conversation_id, [])

    def get_context_string(self, conversation_id: str) -> str:
        """Get conversation context as string."""
        history = self.get_history(conversation_id)
        if not history:
            return ""

        context = "Previous conversation:\n"
        for msg in history[-6:]:  # Last 3 exchanges
            context += f"{msg.role.upper()}: {msg.content}\n"

        return context


# ============================================================================
# Global State
# ============================================================================

class ServerState:
    """Global server state."""
    orchestrator: Optional[WebResearchOrchestrator] = None
    conversation_manager: ConversationManager = None
    audit_trail: Optional[AuditTrail] = None
    config: Optional[Config] = None


state = ServerState()


# ============================================================================
# Lifecycle with async context manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager for server lifecycle."""
    # Startup
    logger.info("Starting HoloLoom Web Research API server...")

    # Load config
    state.config = Config.fused()
    state.config.enable_agentic_reasoning = True

    # Initialize audit trail
    state.audit_trail = AuditTrail(persist_path="./alignment_logs")

    # Initialize conversation manager
    state.conversation_manager = ConversationManager(max_history=10)

    # Create web-enabled orchestrator
    try:
        search_api_key = os.getenv("SERPAPI_KEY")
        search_provider = "serpapi" if search_api_key else "mock"

        if not search_api_key:
            logger.warning("SERPAPI_KEY not set, using mock provider")

        state.orchestrator = await WebResearchOrchestrator.create(
            config=state.config,
            shards=[],
            enable_web_search=True,
            search_provider=search_provider,
            search_api_key=search_api_key,
            audit_trail=state.audit_trail
        )

        logger.info(f"Web research orchestrator ready (provider: {search_provider})")

    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise

    logger.info("Server ready!")

    yield

    # Shutdown
    logger.info("Shutting down server...")
    if state.orchestrator:
        await state.orchestrator.close()


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="HoloLoom Web Research API",
    description="Perplexity-style web research with Matryoshka filtering",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "HoloLoom Web Research API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "web_search_enabled": state.orchestrator is not None
    }


@app.post("/research/web", response_model=WebResearchResponse)
async def research_web_endpoint(request: WebResearchRequest):
    """
    Web-enhanced research with citations.

    Performs Perplexity-style research:
    1. Search web with Matryoshka filtering
    2. Convert to memory shards
    3. Synthesize with citations
    4. Return complete result

    Example:
        POST /research/web
        {
          "query": "What is Thompson Sampling?",
          "max_web_results": 10,
          "enable_citations": true
        }
    """
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Get conversation context if provided
        context_prefix = ""
        if request.conversation_id:
            context = state.conversation_manager.get_context_string(request.conversation_id)
            if context:
                context_prefix = context + "\n\nCurrent query: "

        # Add conversation context to query
        full_query = context_prefix + request.query

        # Execute web research
        result = await state.orchestrator.research_web(
            query=full_query,
            max_web_results=request.max_web_results,
            max_steps=request.max_steps,
            enable_citations=request.enable_citations
        )

        # Store in conversation
        if request.conversation_id:
            state.conversation_manager.add_message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.query
            )
            state.conversation_manager.add_message(
                conversation_id=request.conversation_id,
                role="assistant",
                content=result.cited_response,
                metadata={
                    "web_results": len(result.search_results),
                    "confidence": result.spacetime.confidence
                }
            )

        # Format response
        return WebResearchResponse(
            response=result.spacetime.response,
            cited_response=result.cited_response,
            confidence=result.spacetime.confidence,
            web_results_count=len(result.search_results),
            memory_shards_count=len(result.memory_shards),
            total_duration_ms=result.total_duration_ms,
            web_search_time_ms=result.web_search_time_ms,
            citations=[
                {
                    "index": i + 1,
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet[:200],
                    "score": r.final_score
                }
                for i, r in enumerate(result.search_results)
            ],
            conversation_id=request.conversation_id
        )

    except Exception as e:
        logger.error(f"Research failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/stream")
async def research_stream_endpoint(request: WebResearchRequest):
    """
    Streaming web research with Server-Sent Events (SSE).

    Streams progress updates:
    - Search progress
    - Reasoning steps
    - Final response

    Example:
        POST /research/stream
        {
          "query": "What is Thompson Sampling?",
          "max_web_results": 10
        }

    Response format (SSE):
        data: {"type": "search_start", "query": "..."}
        data: {"type": "search_result", "result": {...}}
        data: {"type": "synthesis_start"}
        data: {"type": "response", "text": "..."}
        data: {"type": "complete", "duration_ms": 1234}
    """
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    async def event_generator():
        """Generate SSE events."""
        try:
            start_time = datetime.now()

            # Event 1: Search start
            yield f"data: {{'type': 'search_start', 'query': '{request.query}'}}\n\n"

            # Execute research (we'll stream events as we go)
            result = await state.orchestrator.research_web(
                query=request.query,
                max_web_results=request.max_web_results,
                max_steps=request.max_steps,
                enable_citations=request.enable_citations
            )

            # Event 2: Search complete
            yield f"data: {{'type': 'search_complete', 'results_count': {len(result.search_results)}}}\n\n"

            # Event 3: Stream citations
            for i, citation in enumerate(result.search_results):
                import json
                citation_data = {
                    "type": "citation",
                    "index": i + 1,
                    "title": citation.title,
                    "url": citation.url
                }
                yield f"data: {json.dumps(citation_data)}\n\n"

            # Event 4: Synthesis start
            yield f"data: {{'type': 'synthesis_start'}}\n\n"

            # Event 5: Stream response (chunk by sentence)
            sentences = result.cited_response.split(". ")
            for sentence in sentences:
                import json
                yield f"data: {json.dumps({'type': 'response_chunk', 'text': sentence + '. '})}\n\n"
                await asyncio.sleep(0.1)  # Small delay for streaming effect

            # Event 6: Complete
            duration = (datetime.now() - start_time).total_seconds() * 1000
            yield f"data: {json.dumps({'type': 'complete', 'duration_ms': duration, 'confidence': result.spacetime.confidence})}\n\n"

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get conversation history.

    Returns all messages in conversation.
    """
    history = state.conversation_manager.get_history(conversation_id)

    return {
        "conversation_id": conversation_id,
        "message_count": len(history),
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            }
            for msg in history
        ]
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation history."""
    if conversation_id in state.conversation_manager.conversations:
        del state.conversation_manager.conversations[conversation_id]
        return {"status": "deleted", "conversation_id": conversation_id}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")


@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    return {
        "orchestrator_ready": state.orchestrator is not None,
        "active_conversations": len(state.conversation_manager.conversations),
        "audit_trail_entries": len(state.audit_trail.logs) if state.audit_trail else 0,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting HoloLoom Web Research API (development mode)...")
    uvicorn.run(
        "HoloLoom.server.web_research_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

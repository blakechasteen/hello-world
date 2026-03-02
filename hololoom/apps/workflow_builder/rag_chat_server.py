import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from datetime import datetime
import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import repository context manager
# Import repository context manager
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from hololoom.memory.repository_context import (
        create_repo_manager,
        RepositoryContextManager,
        AgentQueryContext,
        AccessLevel
    )
    from hololoom.memory.mcp_rag_server import rag_query
except ImportError as e:
    print(f"Warning: Could not import RAG modules: {e}")
    RepositoryContextManager = None
    AgentQueryContext = Any

# Import Eggroll
try:
    from hololoom.eggroll.integration import EggrollIntegration
except ImportError as e:
    print(f"Warning: Could not import Eggroll: {e}")
    EggrollIntegration = None

# Import LLM
try:
    from hololoom.awareness.llm_integration import create_llm
except ImportError as e:
    print(f"Warning: Could not import LLM integration: {e}")
    create_llm = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HoloLoom RAG Chat Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
repo_manager: Optional[RepositoryContextManager] = None
eggroll: Optional[EggrollIntegration] = None
llm: Any = None
agent_contexts: Dict[str, AgentQueryContext] = {}
threads: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Models
# ============================================================================

class RepositoryInfo(BaseModel):
    """Repository information."""
    id: str
    name: str
    description: str
    path: str
    tags: List[str]
    access: str
    file_count: int
    indexed: bool


class ChatMessage(BaseModel):
    """Chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    sender: str
    attached_repos: List[str] = []
    citations: List[Dict] = []
    timestamp: Optional[str] = None
    mode: Optional[str] = "direct"


class ChatRequest(BaseModel):
    """Chat request."""
    message: str
    session_id: str
    attached_repos: List[str] = []
    mode: str = "direct"


class CreateThreadRequest(BaseModel):
    user_id: str
    agent_name: str
    initial_message: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response."""
    message: str
    citations: List[Dict]
    timestamp: str
    mode: str = "direct"


# ============================================================================
# Initialization
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize repository manager and Eggroll on startup."""
    global repo_manager, eggroll, llm

    logger.info("Initializing RAG Chat Server...")

    if RepositoryContextManager:
        # Create repository manager
        repo_manager = await create_repo_manager()
        # ... (Register default repositories logic kept simple for brevity) ...
        logger.info("Repository Manager initialized")
    
    if EggrollIntegration:
        try:
            eggroll = EggrollIntegration()
            logger.info("Eggroll Integration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Eggroll: {e}")

    if create_llm:
        try:
            llm = create_llm("ollama")
            logger.info("Ollama LLM initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")

    logger.info("RAG Chat Server ready!")

# ... (Other endpoints remain unchanged) ...

@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle chat message.
    """
    logger.info(f"Chat request: {request.message[:50]}... Mode: {request.mode}")

    response_text = ""
    citations = []

    if request.mode == "eggroll" and eggroll:
        # Use Eggroll to generate response
        response_text = await eggroll.generate(request.message)
        
    elif llm:
        # Use Standard LLM (RAG or Direct)
        prompt = request.message
        
        # If RAG context is available (simulated for now as we don't have full RAG pipeline here yet)
        # In a real implementation, we would retrieve context from repo_manager here
        
        try:
            llm_resp = await llm.generate(prompt)
            response_text = llm_resp.content
        except Exception as e:
            response_text = f"LLM Error: {e}"
            
    else:
        # Fallback
        response_text = f"Echo ({request.mode}): {request.message} (LLM unavailable)"

    return ChatResponse(
        message=response_text,
        citations=citations,
        timestamp=datetime.now().isoformat(),
        mode=request.mode
    )


# ============================================================================
# WebSocket for Real-Time Chat
# ============================================================================

@app.websocket("/ws/thread/{thread_id}")
async def websocket_thread(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for real-time chat in a thread.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {thread_id}")

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get('message', '')
            mode = data.get('mode', 'direct')
            
            logger.info(f"WS message: {message[:50]}... Mode: {mode}")

            # Send "Thinking" status
            await websocket.send_json({
                'type': 'status',
                'message': 'Thinking...'
            })

            # Process
            response_text = ""
            confidence = 1.0
            
            if mode == "eggroll" and eggroll:
                # Trigger Eggroll
                # 1. Select Pattern
                pattern = eggroll.forced_pattern if eggroll.forced_pattern else eggroll.shuttle.select_pattern()
                
                await websocket.send_json({
                    'type': 'status',
                    'message': f'Selected Strategy: {pattern}...'
                })
                
                # 2. Generate with Eggroll (MirrorCore)
                try:
                    response_text = await eggroll.generate(message)
                except Exception as e:
                    response_text = f"Eggroll Error: {e}"
                
                confidence = 0.95
                
            elif llm:
                # Standard LLM
                try:
                    llm_resp = await llm.generate(message)
                    response_text = llm_resp.content
                except Exception as e:
                    response_text = f"LLM Error: {e}"
            else:
                response_text = f"Echo: {message} (LLM unavailable)"

            # Send response
            await websocket.send_json({
                'type': 'message',
                'content': response_text,
                'confidence': confidence,
                'mode': mode,
                'timestamp': datetime.now().isoformat()
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {thread_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))

@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    await websocket.accept()
    # Keep alive
    try:
        while True:
            await asyncio.sleep(10)
    except Exception:
        pass

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

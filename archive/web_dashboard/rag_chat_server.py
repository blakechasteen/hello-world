from __future__ import annotations
"""
RAG-Powered Chat Server with Repository Context
================================================
Backend server for the chat_with_rag.html UI.

Features:
- Repository management API
- Chat with repository context
- Real-time RAG queries
- Citation tracking

Usage:
    python rag_chat_server.py

Then open: http://localhost:8002/chat

Author: HoloLoom Team
Date: 2025-11-04
"""

import logging
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# Import repository context manager
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from hololoom.memory.mcp_rag_server import rag_query
    from hololoom.memory.repository_context import (
        AccessLevel,
        AgentQueryContext,
        RepositoryContextManager,
        create_repo_manager,
    )
except ImportError as e:
    print(f"Warning: Could not import HoloLoom modules: {e}")
    RepositoryContextManager = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HoloLoom RAG Chat Server")

# Global state
repo_manager: RepositoryContextManager | None = None
agent_contexts: dict[str, AgentQueryContext] = {}


# ============================================================================
# Models
# ============================================================================

class RepositoryInfo(BaseModel):
    """Repository information."""
    id: str
    name: str
    description: str
    path: str
    tags: list[str]
    access: str
    file_count: int
    indexed: bool


class ChatMessage(BaseModel):
    """Chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    sender: str
    attached_repos: list[str] = []
    citations: list[dict] = []
    timestamp: str | None = None


class ChatRequest(BaseModel):
    """Chat request."""
    message: str
    session_id: str
    attached_repos: list[str] = []


class ChatResponse(BaseModel):
    """Chat response."""
    message: str
    citations: list[dict]
    timestamp: str


# ============================================================================
# Initialization
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize repository manager on startup."""
    global repo_manager

    logger.info("Initializing RAG Chat Server...")

    if RepositoryContextManager is None:
        logger.warning("Repository context manager not available - running in demo mode")
        return

    # Create repository manager
    repo_manager = await create_repo_manager()

    # Register default repositories
    base_path = Path(__file__).parent.parent.parent

    try:
        await repo_manager.add_repository(
            name="hololoom",
            path=str(base_path / "hololoom"),
            tags={"python", "ml", "core"},
            access_level=AccessLevel.PUBLIC,
            description="Core ML system",
            auto_index=False  # Set to True for production
        )

        await repo_manager.add_repository(
            name="squad",
            path=str(base_path / "squad"),
            tags={"typescript", "vscode", "frontend"},
            access_level=AccessLevel.PUBLIC,
            description="VS Code extension",
            auto_index=False
        )

        await repo_manager.add_repository(
            name="cos",
            path=str(base_path / "cos"),
            tags={"business", "private"},
            access_level=AccessLevel.PRIVATE,
            description="Business planning",
            auto_index=False
        )

        await repo_manager.add_repository(
            name="dashboard",
            path=str(base_path / "dashboard"),
            tags={"react", "frontend"},
            access_level=AccessLevel.PUBLIC,
            description="Web dashboard",
            auto_index=False
        )

        logger.info(f"Registered {len(repo_manager.repositories)} repositories")

    except Exception as e:
        logger.error(f"Failed to register repositories: {e}")

    logger.info("RAG Chat Server ready!")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Redirect to chat UI."""
    return {"message": "HoloLoom RAG Chat Server", "ui": "/chat"}


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """Serve chat UI."""
    ui_path = Path(__file__).parent / "chat_with_rag.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="Chat UI not found")
    return FileResponse(ui_path)


@app.get("/api/repositories")
async def list_repositories() -> list[RepositoryInfo]:
    """List all registered repositories."""
    if repo_manager is None:
        # Demo mode
        return [
            RepositoryInfo(
                id="hololoom",
                name="hololoom",
                description="Core ML system",
                path="c:/Users/blake/OneDrive/Documents/mythRL/HoloLoom",
                tags=["python", "ml", "core"],
                access="public",
                file_count=156,
                indexed=False
            ),
            RepositoryInfo(
                id="squad",
                name="squad",
                description="VS Code extension",
                path="c:/Users/blake/OneDrive/Documents/mythRL/squad",
                tags=["typescript", "vscode", "frontend"],
                access="public",
                file_count=42,
                indexed=False
            ),
            RepositoryInfo(
                id="cos",
                name="cos",
                description="Business planning",
                path="c:/Users/blake/OneDrive/Documents/mythRL/cos",
                tags=["business", "private"],
                access="private",
                file_count=18,
                indexed=False
            )
        ]

    repos = []
    for repo in repo_manager.repositories.values():
        repos.append(RepositoryInfo(
            id=repo.name,
            name=repo.name,
            description=repo.description,
            path=repo.path,
            tags=list(repo.tags),
            access=repo.access_level.value,
            file_count=repo.file_count,
            indexed=repo.indexed_at is not None
        ))

    return repos


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle chat message with repository context.

    Args:
        request: Chat request with message and attached repos

    Returns:
        Chat response with answer and citations
    """
    logger.info(f"Chat request: {request.message[:50]}... with repos: {request.attached_repos}")

    if repo_manager is None:
        # Demo mode
        return ChatResponse(
            message=f"[Demo Mode] Response to: {request.message}",
            citations=[],
            timestamp=datetime.now().isoformat()
        )

    # Create agent context for this session
    session_id = request.session_id
    if session_id not in agent_contexts:
        agent_contexts[session_id] = repo_manager.create_agent_context(
            agent_id=session_id,
            allowed_repos=set(request.attached_repos) if request.attached_repos else None,
            access_level=AccessLevel.PUBLIC
        )

    agent_context = agent_contexts[session_id]

    # Query with RAG
    try:
        results = await agent_context.query(
            query_text=request.message,
            limit=5
        )

        # Format citations
        citations = [
            {
                'repo': res['context']['repository'],
                'file': res['context']['file_path'],
                'entity': res['context'].get('entity_name', 'N/A'),
                'score': res['score']
            }
            for res in results
        ]

        # Generate response (in production, use LLM)
        if len(results) == 0:
            response_text = (
                "I don't have any relevant code context from the attached repositories. "
                "Try attaching more repositories or rephrasing your question."
            )
        else:
            # Simulate response generation
            repo_names = list(set(c['repo'] for c in citations))
            response_text = (
                f"Based on code from <strong>{', '.join(repo_names)}</strong>:<br><br>"
                f"I found {len(results)} relevant code snippets. "
                f"[In production, this would use an LLM to synthesize an answer from the code context]"
            )

        return ChatResponse(
            message=response_text,
            citations=citations,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/repositories/{repo_id}/index")
async def index_repository(repo_id: str):
    """
    Index a repository for RAG.

    Args:
        repo_id: Repository ID to index

    Returns:
        Status message
    """
    if repo_manager is None:
        raise HTTPException(status_code=503, detail="Repository manager not available")

    if repo_id not in repo_manager.repositories:
        raise HTTPException(status_code=404, detail=f"Repository not found: {repo_id}")

    logger.info(f"Indexing repository: {repo_id}")

    try:
        await repo_manager.index_repository(repo_id)
        repo = repo_manager.repositories[repo_id]

        return {
            "status": "success",
            "repo_id": repo_id,
            "files_indexed": repo.file_count,
            "lines_indexed": repo.line_count
        }

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "repo_manager": repo_manager is not None,
        "repositories": len(repo_manager.repositories) if repo_manager else 0,
        "active_sessions": len(agent_contexts)
    }


# ============================================================================
# WebSocket for Real-Time Chat (Optional)
# ============================================================================

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.

    Args:
        websocket: WebSocket connection
        session_id: User session ID
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get('message', '')
            attached_repos = data.get('attached_repos', [])

            logger.info(f"WS message: {message[:50]}... from {session_id}")

            # Process with RAG
            response = await chat(ChatRequest(
                message=message,
                session_id=session_id,
                attached_repos=attached_repos
            ))

            # Send response
            await websocket.send_json({
                'type': 'response',
                'message': response.message,
                'citations': response.citations,
                'timestamp': response.timestamp
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        if session_id in agent_contexts:
            del agent_contexts[session_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002

    print(f"\n{'='*60}")
    print("HoloLoom RAG Chat Server")
    print(f"{'='*60}")
    print(f"\n🌐 Chat UI:  http://localhost:{port}/chat")
    print(f"📖 API Docs: http://localhost:{port}/docs")
    print(f"🔍 Health:   http://localhost:{port}/api/health")
    print(f"\n{'='*60}\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

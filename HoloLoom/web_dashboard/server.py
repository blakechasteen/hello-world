"""
Unified Multithreaded Chat - FastAPI Server
============================================

WebSocket-based server for real-time chat with awareness.

Endpoints:
- GET / - Dashboard HTML
- WS /ws - WebSocket for real-time chat
- GET /api/threads - List all threads
- GET /api/thread/{id} - Get specific thread

Usage:
    python HoloLoom/web_dashboard/server.py

Then open: http://localhost:8000
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import asyncio
from typing import Dict, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from HoloLoom.awareness import (
        CompositionalAwarenessLayer,
        DualStreamGenerator,
        MetaAwarenessLayer,
        OllamaLLM,
        LLM_AVAILABLE
    )
    AWARENESS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Awareness layer not available: {e}")
    AWARENESS_AVAILABLE = False

# Create FastAPI app
app = FastAPI(title="HoloLoom Multithreaded Chat")

# Active WebSocket connections
active_connections: List[WebSocket] = []

# Thread manager (will be initialized on startup)
thread_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize thread manager on startup"""
    global thread_manager

    # Import here to avoid circular dependency
    from HoloLoom.web_dashboard.thread_manager import ThreadManager

    # Initialize persistent memory backend (optional)
    memory_backend = None
    try:
        from HoloLoom.config import Config, MemoryBackend
        from HoloLoom.memory.backend_factory import create_memory_backend

        config = Config.fast()
        config.memory_backend = MemoryBackend.HYBRID  # Neo4j + Qdrant with fallback

        memory_backend = await create_memory_backend(config)
        print("✓ Persistent memory backend initialized (HYBRID)")
    except Exception as e:
        print(f"⚠ Memory backend unavailable - running ephemeral: {e}")
        memory_backend = None

    # Initialize awareness layer if available
    if AWARENESS_AVAILABLE:
        awareness = CompositionalAwarenessLayer()

        # Try to initialize Ollama
        try:
            llm = OllamaLLM(model="llama3.2:3b")
            if llm.is_available():
                print("✓ Ollama LLM available")
                dual_stream_gen = DualStreamGenerator(awareness, llm_generator=llm)
            else:
                print("⚠ Ollama not running - using templates")
                dual_stream_gen = DualStreamGenerator(awareness, llm_generator=None)
        except:
            print("⚠ Ollama error - using templates")
            dual_stream_gen = DualStreamGenerator(awareness, llm_generator=None)

        thread_manager = ThreadManager(
            awareness_layer=awareness,
            llm_generator=dual_stream_gen,
            memory_backend=memory_backend
        )

        if memory_backend:
            print("✓ Thread manager initialized with awareness + persistent memory")
        else:
            print("✓ Thread manager initialized with awareness (ephemeral)")
    else:
        thread_manager = ThreadManager(
            awareness_layer=None,
            llm_generator=None,
            memory_backend=memory_backend
        )
        print("✓ Thread manager initialized (no awareness)")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve main dashboard HTML"""
    html_file = Path(__file__).parent / "index.html"

    if html_file.exists():
        return FileResponse(html_file)
    else:
        # Return basic HTML if file doesn't exist yet
        return HTMLResponse(content=get_basic_html())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            action = message_data.get('action', 'send_message')

            if action == 'send_message':
                # Process user message
                user_input = message_data.get('message', '')
                explicit_thread_id = message_data.get('thread_id')

                # Process with thread manager
                result = await thread_manager.process_message(
                    user_input,
                    explicit_thread_id=explicit_thread_id
                )

                # Send response back to client
                await websocket.send_json({
                    'type': 'message_response',
                    'data': result
                })

            elif action == 'get_threads':
                # Return all threads
                threads = {
                    thread_id: thread.to_dict()
                    for thread_id, thread in thread_manager.threads.items()
                }

                await websocket.send_json({
                    'type': 'threads_list',
                    'data': threads
                })

            elif action == 'get_thread':
                # Return specific thread
                thread_id = message_data.get('thread_id')
                if thread_id in thread_manager.threads:
                    thread = thread_manager.threads[thread_id]
                    await websocket.send_json({
                        'type': 'thread_detail',
                        'data': thread.to_dict()
                    })

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"Client disconnected. Active connections: {len(active_connections)}")


@app.get("/api/threads")
async def get_threads():
    """Get all conversation threads"""
    if thread_manager:
        threads = {
            thread_id: thread.to_dict()
            for thread_id, thread in thread_manager.threads.items()
        }
        return {'threads': threads}
    return {'threads': {}}


@app.get("/api/thread/{thread_id}")
async def get_thread(thread_id: str):
    """Get specific thread"""
    if thread_manager and thread_id in thread_manager.threads:
        thread = thread_manager.threads[thread_id]
        return thread.to_dict()
    return {'error': 'Thread not found'}


@app.get("/api/status")
async def get_status():
    """Get server status"""
    return {
        'status': 'running',
        'awareness_available': AWARENESS_AVAILABLE,
        'llm_available': LLM_AVAILABLE if AWARENESS_AVAILABLE else False,
        'active_connections': len(active_connections),
        'total_threads': thread_manager.thread_count if thread_manager else 0,
        'total_messages': thread_manager.total_messages if thread_manager else 0,
    }


def get_basic_html() -> str:
    """Return basic HTML if index.html doesn't exist"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>HoloLoom - Multithreaded Chat</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        .status {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .success { color: #00ff88; }
        .warning { color: #ffaa00; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧵 HoloLoom - Unified Multithreaded Chat</h1>
        <div class="status">
            <h2>Server Status</h2>
            <p class="success">✓ Server running on http://localhost:8000</p>
            <p class="warning">⚠ Frontend UI not yet created</p>
            <p>Creating <code>index.html</code> next...</p>
        </div>
        <div class="status">
            <h2>API Endpoints</h2>
            <ul>
                <li><code>GET /api/status</code> - Server status</li>
                <li><code>GET /api/threads</code> - List threads</li>
                <li><code>WS /ws</code> - WebSocket chat</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("  HoloLoom - Unified Multithreaded Chat Server")
    print("="*60)
    print()
    print("Starting server...")
    print("Dashboard: http://localhost:8000")
    print("WebSocket: ws://localhost:8000/ws")
    print()
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)

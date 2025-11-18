"""
WebSocket Server Main Entry Point

Run this separately from the main API server:
    uvicorn websocket_main:app --port 8001 --reload

Or combine with main API using multiple workers
"""
from websocket.server import socket_app
import uvicorn

# The ASGI app for WebSocket server
app = socket_app

if __name__ == "__main__":
    uvicorn.run(
        "websocket_main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )

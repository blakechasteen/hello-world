#!/usr/bin/env python3
"""
HoloLoom Studio Server
======================
The unified entry point for the HoloLoom Studio standalone application.
Hosts the Agentic API, Agent Manager API, and the React Dashboard.
"""

import os
import logging
import webbrowser
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import existing APIs
from HoloLoom.apps.server.agentic_api import app as agentic_app
from HoloLoom.apps.server.agent_manager_api import app as manager_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the main app
app = FastAPI(
    title="HoloLoom Studio",
    description="The local control plane for HoloLoom Agents",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the APIs
app.mount("/api/agentic", agentic_app)
app.mount("/api/manager", manager_app)

# Path to static assets
# Assuming build output is in HoloLoom/dist/ (set in vite.config.js)
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "dist"

@app.get("/health")
async def health():
    return {"status": "ok", "app": "HoloLoom Studio"}

# Serve static files if they exist
if STATIC_DIR.exists():
    logger.info(f"Serving static files from {STATIC_DIR}")
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    
    # Catch-all for React Router
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(status_code=404, content={"message": "Not found"})
else:
    logger.warning(f"⚠️ Static directory {STATIC_DIR} not found. Dashboard will not be available.")
    logger.warning("Run 'cd HoloLoom/hololoom-dashboard && npm install && npm run build' to generate assets.")

    @app.get("/")
    async def root():
        return {
            "message": "HoloLoom Studio API is running, but the Dashboard assets were not found.",
            "instructions": "Build the dashboard using 'npm run build' in the hololoom-dashboard directory."
        }

def start_studio(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    """Start the Studio server."""
    if open_browser:
        # Give the server a moment to start before opening browser
        import threading
        import time
        
        def open_url():
            time.sleep(1.5)
            url = f"http://{host}:{port}"
            logger.info(f"Opening Studio in browser: {url}")
            webbrowser.open(url)
            
        threading.Thread(target=open_url, daemon=True).start()
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_studio()

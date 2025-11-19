"""
The Rusty Mug Tavern - FastAPI Server

Serves the game frontend and handles API requests.
Routes player actions to game engine and manages session state.

Created: 2025-11-16
"""

import os
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from game_engine import TavernGameEngine


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="The Rusty Mug Tavern",
    description="An Elle Game Engine Demo - Interactive Text Adventure",
    version="1.0.0"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global game engine instance (one per server process)
game_engines: Dict[str, TavernGameEngine] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class NewGameRequest(BaseModel):
    """Request to start new game."""
    player_name: Optional[str] = "Traveler"


class ActionRequest(BaseModel):
    """Request to perform action."""
    session_id: str
    action_type: str
    params: Dict[str, Any] = {}


class SaveGameRequest(BaseModel):
    """Request to save game."""
    session_id: str
    save_name: Optional[str] = "quicksave"


class LoadGameRequest(BaseModel):
    """Request to load game."""
    save_name: str


# ============================================================================
# Helper Functions
# ============================================================================

async def get_or_create_engine(session_id: Optional[str] = None) -> TavernGameEngine:
    """Get existing engine or create new one."""
    if session_id and session_id in game_engines:
        return game_engines[session_id]

    # Create new engine
    engine = TavernGameEngine(elle_api_url=os.getenv("ELLE_API_URL", "http://localhost:8000"))
    await engine.__aenter__()  # Initialize HTTP session

    if session_id:
        game_engines[session_id] = engine

    return engine


async def cleanup_engine(session_id: str):
    """Cleanup and remove engine."""
    if session_id in game_engines:
        engine = game_engines[session_id]
        await engine.__aexit__(None, None, None)
        del game_engines[session_id]


# ============================================================================
# Static File Routes
# ============================================================================

# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main game page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")

    with open(index_path, 'r') as f:
        return HTMLResponse(content=f.read())


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon if it exists."""
    favicon_path = STATIC_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404)


# ============================================================================
# Game API Routes
# ============================================================================

@app.post("/api/game/new")
async def new_game(request: NewGameRequest) -> JSONResponse:
    """
    Start a new game.

    Returns:
        Initial game state with session ID
    """
    try:
        # Create new engine
        engine = await get_or_create_engine()

        # Initialize game
        result = await engine.initialize()

        # Store engine with session ID
        session_id = result["game_state"]["session_id"]
        game_engines[session_id] = engine

        # Set player name if provided
        if request.player_name and request.player_name != "Traveler":
            engine.game_state.player.name = request.player_name

        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "message": f"Welcome to The Rusty Mug, {request.player_name}!",
            "game_state": engine.get_public_state()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create game: {str(e)}")


@app.post("/api/game/action")
async def process_action(request: ActionRequest) -> JSONResponse:
    """
    Process a player action.

    Actions:
    - talk: Talk to NPC (params: npc_id, player_message)
    - look: Look around current location
    - move: Move to different location (params: destination)
    - quest_accept: Accept a quest (params: quest_id)
    - quest_complete: Complete a quest (params: quest_id)
    - give_gift: Give item/gold to NPC (params: npc_id, gift_type, amount)

    Returns:
        Action result and updated game state
    """
    try:
        # Get engine for session
        engine = game_engines.get(request.session_id)
        if not engine:
            raise HTTPException(status_code=404, detail="Session not found")

        # Process action
        result = await engine.process_player_action(
            request.action_type,
            **request.params
        )

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action failed: {str(e)}")


@app.post("/api/game/save")
async def save_game(request: SaveGameRequest) -> JSONResponse:
    """
    Save the current game state.

    Returns:
        Save confirmation with file path
    """
    try:
        # Get engine for session
        engine = game_engines.get(request.session_id)
        if not engine:
            raise HTTPException(status_code=404, detail="Session not found")

        # Create saves directory
        saves_dir = BASE_DIR / "saves"
        saves_dir.mkdir(exist_ok=True)

        # Save game
        save_file = str(saves_dir / f"{request.save_name}.json")
        result = await engine.save_game(save_file)

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@app.post("/api/game/load")
async def load_game(request: LoadGameRequest) -> JSONResponse:
    """
    Load a saved game.

    Returns:
        Game state from save file
    """
    try:
        # Create new engine
        engine = await get_or_create_engine()

        # Load game
        saves_dir = BASE_DIR / "saves"
        save_file = str(saves_dir / f"{request.save_name}.json")

        result = await engine.load_game(save_file)

        if result["status"] == "success":
            # Store engine with session ID
            session_id = result["game_state"]["session_id"]
            game_engines[session_id] = engine

            return JSONResponse({
                "status": "success",
                "session_id": session_id,
                "message": result["message"],
                "game_state": result["game_state"]
            })
        else:
            await cleanup_engine(None)  # Clean up unused engine
            return JSONResponse(result, status_code=404)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")


@app.get("/api/game/saves")
async def list_saves() -> JSONResponse:
    """
    List all available save files.

    Returns:
        List of save file names
    """
    try:
        saves_dir = BASE_DIR / "saves"
        if not saves_dir.exists():
            return JSONResponse({"saves": []})

        saves = [
            {
                "name": f.stem,
                "file": f.name,
                "modified": f.stat().st_mtime
            }
            for f in saves_dir.glob("*.json")
        ]

        # Sort by modification time (newest first)
        saves.sort(key=lambda x: x["modified"], reverse=True)

        return JSONResponse({"saves": saves})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list saves: {str(e)}")


@app.delete("/api/game/session/{session_id}")
async def end_session(session_id: str) -> JSONResponse:
    """
    End a game session and cleanup resources.

    Returns:
        Cleanup confirmation
    """
    try:
        await cleanup_engine(session_id)
        return JSONResponse({
            "status": "success",
            "message": "Session ended"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# ============================================================================
# WebSocket for Real-Time Updates (Optional)
# ============================================================================

active_connections: Dict[str, WebSocket] = {}


@app.websocket("/ws/game/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket connection for real-time game updates.

    Use this for future features like:
    - Multiplayer chat
    - Live NPC behaviors
    - World events
    """
    await websocket.accept()
    active_connections[session_id] = websocket

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()

            # Echo back for now (implement real-time features later)
            await websocket.send_json({
                "type": "echo",
                "data": data
            })

    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "The Rusty Mug Tavern",
        "active_sessions": len(game_engines)
    }


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("🍺 The Rusty Mug Tavern is opening...")
    print(f"📁 Static files: {STATIC_DIR}")

    # Create directories
    (BASE_DIR / "saves").mkdir(exist_ok=True)

    print("✅ Server ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🍺 The Rusty Mug is closing for the night...")

    # Cleanup all active engines
    for session_id in list(game_engines.keys()):
        await cleanup_engine(session_id)

    print("✅ All sessions closed.")


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🍺 The Rusty Mug Tavern - Elle Game Engine Demo")
    print("=" * 60)
    print()
    print("Starting server on http://localhost:8001")
    print("Open your browser and navigate to the URL above!")
    print()
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

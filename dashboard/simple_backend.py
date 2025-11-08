#!/usr/bin/env python3
"""
🔥 Simplified mythRL Dashboard Backend (Dark Fire Edition)
===========================================================
Uses HoloLoom unified API for narrative analysis.
"""

import asyncio
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add mythRL to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.unified_api import HoloLoom
from HoloLoom.matryoshka_interpreter import MatryoshkaInterpreter
from HoloLoom.visualization import (
    MatryoshkaAnalysis,
    StreamOfThought,
    ResonanceVisualization,
    WarpSpaceVisualization,
    ConvergenceVisualization
)

# Try to import analysis modules
try:
    from HoloLoom.resonance.shed import ResonanceShed
    from HoloLoom.warp.space import WarpSpace
    from HoloLoom.convergence.engine import ConvergenceEngine
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Analysis modules not fully available: {e}")
    ANALYSIS_MODULES_AVAILABLE = False


app = FastAPI(title="🔥 mythRL Dark Fire API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global HoloLoom instance and interpreters
loom_instance = None
matryoshka_interpreter = None


class AnalyzeRequest(BaseModel):
    text: str
    domain: str = "mythology"


class AnalyzeResponse(BaseModel):
    domain: str
    complexity: float
    confidence: float
    max_depth: str
    deepest_meaning: str
    characters: list
    current_stage: Optional[str] = None


@app.on_event("startup")
async def startup():
    """Initialize HoloLoom on startup."""
    global loom_instance, matryoshka_interpreter
    print("🔥 Initializing HoloLoom...")
    loom_instance = await HoloLoom.create(pattern="fast")
    matryoshka_interpreter = MatryoshkaInterpreter()
    print("✅ HoloLoom ready!")
    print("✅ Matryoshka interpreter loaded!")
    if ANALYSIS_MODULES_AVAILABLE:
        print("✅ Analysis modules available!")


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "🔥 mythRL Dark Fire API",
        "version": "1.0.0",
        "status": "blazing",
        "endpoints": {
            "analyze": "/api/analyze",
            "domains": "/api/domains",
            "examples": "/api/examples",
            "docs": "/docs"
        }
    }


@app.get("/api/domains")
async def get_domains():
    """Get available narrative domains."""
    return [
        {"id": "mythology", "name": "Mythology", "color": "#FF4500"},
        {"id": "business", "name": "Business", "color": "#FF6347"},
        {"id": "science", "name": "Science", "color": "#FF8C00"},
        {"id": "personal", "name": "Personal", "color": "#FFA500"},
        {"id": "product", "name": "Product", "color": "#FF7F50"},
        {"id": "history", "name": "History", "color": "#FFD700"}
    ]


@app.post("/api/analyze")
async def analyze_narrative(request: AnalyzeRequest):
    """Analyze narrative text."""
    global loom_instance
    
    if loom_instance is None:
        loom_instance = await HoloLoom.create(pattern="fast")
    
    try:
        # Query HoloLoom
        result = await loom_instance.query(request.text)
        
        # Extract spacetime data
        spacetime = result if hasattr(result, 'content') else result
        
        # Parse response (HoloLoom returns natural language)
        return {
            "domain": request.domain,
            "complexity": 0.65,  # Mock for now
            "confidence": 0.85,
            "max_depth": "ARCHETYPAL",
            "deepest_meaning": str(spacetime.content if hasattr(spacetime, 'content') else spacetime),
            "characters": [],
            "current_stage": None,
            "base_analysis": {
                "text": request.text,
                "response": str(spacetime)
            }
        }
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return {
            "domain": request.domain,
            "complexity": 0.5,
            "confidence": 0.7,
            "max_depth": "SYMBOLIC",
            "deepest_meaning": f"Analysis in progress... {request.text[:100]}",
            "characters": [],
            "current_stage": None,
            "base_analysis": {"error": str(e)}
        }


@app.get("/api/examples")
async def get_examples():
    """Get example narratives for each domain."""
    return {
        "mythology": "Odysseus, guided by Athena, faced the Cyclops and overcame his pride. The journey home transformed him from warrior to wise king.",
        "business": "Sarah quit her corporate job to build a startup. Her advisor warned: 'You'll pivot three times.' Months of failures followed, then one customer email changed everything.",
        "science": "Dr. Chen's experiment contradicted 50 years of theory. Her PI dismissed it as error. But after three replications, they couldn't ignore the paradigm shift.",
        "personal": "In therapy, I finally faced what I'd avoided for years. My inner critic screamed, but as I sat with discomfort, the wound became a doorway.",
        "product": "User interviews revealed we'd solved the wrong problem. The team resisted redesigning. But we fell in love with the problem, and users fell in love with our solution.",
        "history": "The protesters gathered despite warnings. Each crackdown spawned ten more protests. When the masses flooded the capital, the old order crumbled."
    }


# Simple loop mode stubs for UI compatibility
@app.post("/api/loop/start")
async def start_loop():
    """Start loop mode (stub)."""
    return {"status": "started", "mode": "batch", "message": "🔥 Loop mode ready"}


@app.post("/api/loop/stop")
async def stop_loop():
    """Stop loop mode (stub)."""
    return {"status": "stopped", "message": "Loop halted"}


@app.get("/api/loop/status")
async def loop_status():
    """Get loop status (stub)."""
    return {
        "active": False,
        "stats": {"processed": 0, "queued": 0, "rate": 0.0, "avgTime": 0.0}
    }


@app.post("/api/loop/add-task")
async def add_task():
    """Add task to loop (stub)."""
    return {"status": "queued", "message": "Task added to queue"}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🔥 Starting mythRL Dark Fire API Server")
    print("="*70)
    print("📡 http://localhost:8000")
    print("📊 Docs: http://localhost:8000/docs")
    print("🔥 Dark Mode: ENGAGED")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

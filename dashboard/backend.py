#!/usr/bin/env python3
"""
🚀 FastAPI Backend for mythRL Dashboard (Dark Fire Edition)
============================================================
Real-time narrative intelligence API using HoloLoom unified API.
"""

import asyncio
import time
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add mythRL to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.unified_api import HoloLoom


app = FastAPI(title="mythRL Narrative Intelligence API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global HoloLoom instance
loom_instance = None


class AnalyzeRequest(BaseModel):
    text: str
    domain: str = "mythology"
    auto_detect: bool = False


class AnalyzeResponse(BaseModel):
    domain: str
    base_analysis: dict
    domain_translation: dict
    insights: dict


@app.get("/")
async def root():
    """API health check."""
    return {
        "status": "online",
        "service": "mythRL Narrative Intelligence",
        "version": "1.0.0",
        "features": [
            "cross-domain-adaptation",
            "real-time-streaming",
            "matryoshka-depth-analysis",
            "campbell-journey-mapping"
        ]
    }


@app.get("/api/domains")
async def list_domains():
    """List all available narrative domains."""
    domains = cross_domain_adapter.list_domains()
    domain_info = []
    
    for domain_name in domains:
        info = cross_domain_adapter.get_domain_info(domain_name)
        if info:
            domain_info.append(info)
    
    return {
        "domains": domain_info,
        "count": len(domain_info)
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_narrative(request: AnalyzeRequest):
    """
    Analyze narrative with domain-specific adaptation.
    
    Returns complete depth analysis with domain translation.
    """
    # Map domain string to enum
    domain_map = {
        'mythology': NarrativeDomain.MYTHOLOGY,
        'business': NarrativeDomain.BUSINESS,
        'science': NarrativeDomain.SCIENCE,
        'personal': NarrativeDomain.PERSONAL,
        'product': NarrativeDomain.PRODUCT,
        'history': NarrativeDomain.HISTORY
    }
    
    if request.auto_detect:
        result = await cross_domain_adapter.analyze_with_domain(
            request.text,
            auto_detect=True
        )
    else:
        domain = domain_map.get(request.domain, NarrativeDomain.MYTHOLOGY)
        result = await cross_domain_adapter.analyze_with_domain(
            request.text,
            domain=domain
        )
    
    return AnalyzeResponse(**result)


@app.websocket("/ws/narrative")
async def narrative_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming narrative analysis.
    
    Client sends text chunks, server sends streaming events:
    - chunk_added
    - complexity_update
    - gate_unlocked
    - character_detected
    - stage_transition
    - narrative_shift
    - analysis_complete
    """
    await websocket.accept()
    
    try:
        # Create streaming analyzer
        analyzer = StreamingNarrativeAnalyzer(
            chunk_size=50,
            update_interval=0.5,
            enable_shift_detection=True
        )
        
        # Register callback to send events
        async def send_event(event):
            await websocket.send_json({
                'event_type': event.event_type.value,
                'timestamp': event.timestamp,
                'data': event.data,
                'cumulative_text_length': event.cumulative_text_length
            })
        
        analyzer.on_event(send_event)
        
        # Text stream from WebSocket
        async def text_stream():
            while True:
                data = await websocket.receive_text()
                if data == "END":
                    break
                yield data
        
        # Run streaming analysis
        async for event in analyzer.stream_analyze(text_stream()):
            pass  # Events sent via callback
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


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


# ============================================================================
# LOOP MODE ENDPOINTS
# ============================================================================

class LoopStartRequest(BaseModel):
    mode: str = "batch"  # batch, continuous, scheduled
    rate_limit: int = 5  # tasks per second
    auto_detect: bool = True


class LoopTaskRequest(BaseModel):
    text: str
    domain: Optional[str] = None
    priority: str = "normal"  # low, normal, high, urgent


@app.post("/api/loop/start")
async def start_loop(request: LoopStartRequest):
    """Start the narrative loop engine."""
    global loop_engine, loop_task, loop_stats
    
    if loop_engine is not None:
        return {"status": "already_running", "message": "Loop is already active"}
    
    # Map mode string to enum
    mode_map = {
        "batch": LoopMode.BATCH,
        "continuous": LoopMode.CONTINUOUS,
        "scheduled": LoopMode.SCHEDULED
    }
    
    loop_mode = mode_map.get(request.mode, LoopMode.BATCH)
    
    # Create loop engine
    loop_engine = NarrativeLoopEngine(
        mode=loop_mode,
        rate_limit=request.rate_limit
    )
    
    # Stats callback
    def update_stats(stats_obj):
        loop_stats["processed"] = stats_obj.tasks_processed
        loop_stats["queued"] = len(loop_engine.task_queue)
        loop_stats["rate"] = stats_obj.tasks_per_second
        loop_stats["avgTime"] = stats_obj.average_time_ms
    
    # Register callback
    loop_engine.on_result = lambda task, result: update_stats(loop_engine.stats)
    
    # Start loop in background
    loop_task = asyncio.create_task(loop_engine.run())
    
    return {
        "status": "started",
        "mode": request.mode,
        "rate_limit": request.rate_limit,
        "auto_detect": request.auto_detect
    }


@app.post("/api/loop/stop")
async def stop_loop():
    """Stop the narrative loop engine."""
    global loop_engine, loop_task, loop_stats
    
    if loop_engine is None:
        return {"status": "not_running", "message": "Loop is not active"}
    
    # Stop the loop
    loop_engine.stop()
    
    if loop_task:
        await loop_task
    
    # Reset
    loop_engine = None
    loop_task = None
    loop_stats = {"processed": 0, "queued": 0, "rate": 0.0, "avgTime": 0.0}
    
    return {"status": "stopped"}


@app.get("/api/loop/status")
async def get_loop_status():
    """Get current loop engine status."""
    global loop_engine, loop_stats
    
    if loop_engine is None:
        return {
            "active": False,
            "stats": {"processed": 0, "queued": 0, "rate": 0.0, "avgTime": 0.0}
        }
    
    # Update queue count
    loop_stats["queued"] = len(loop_engine.task_queue)
    
    return {
        "active": True,
        "mode": loop_engine.mode.name.lower(),
        "stats": loop_stats
    }


@app.post("/api/loop/add-task")
async def add_loop_task(request: LoopTaskRequest):
    """Add a task to the loop queue."""
    global loop_engine
    
    if loop_engine is None:
        return {"status": "error", "message": "Loop is not running"}
    
    # Map priority string to enum
    priority_map = {
        "low": Priority.LOW,
        "normal": Priority.NORMAL,
        "high": Priority.HIGH,
        "urgent": Priority.URGENT
    }
    
    priority = priority_map.get(request.priority, Priority.NORMAL)
    
    # Add task
    task_id = f"task_{int(time.time() * 1000)}"
    loop_engine.add_task(
        task_id=task_id,
        text=request.text,
        domain=request.domain,
        priority=priority
    )
    
    return {
        "status": "added",
        "priority": request.priority,
        "queue_size": len(loop_engine.task_queue)
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting mythRL API Server")
    print("📡 http://localhost:8000")
    print("📊 Docs: http://localhost:8000/docs")
    print("🌊 WebSocket: ws://localhost:8000/ws/narrative")
    print("🔁 Loop Mode: /api/loop/*")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
